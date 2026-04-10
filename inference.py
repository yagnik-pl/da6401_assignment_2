"""Inference, evaluation, and report-visualization helpers."""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from common import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    blend_mask,
    box_iou_xywh,
    denormalize_image,
    dice_score,
    draw_boxes,
    pixel_accuracy,
    resolve_path,
)
from data.pets_dataset import OxfordIIITPetDataset
from models import MultiTaskPerceptionModel, VGG11Classifier, VGG11Localizer, VGG11UNet

try:
    import wandb
except ImportError:
    wandb = None


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from '{value}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Assignment 2 inference and report visualizations")
    parser.add_argument("--task", type=str, default="multitask", choices=["classification", "localization", "segmentation", "multitask"])
    parser.add_argument(
        "--mode",
        type=str,
        default="image",
        choices=["dataset", "image", "feature_maps", "activation_hist", "bbox_table", "mask_gallery", "showcase"],
    )
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--data_root", type=str, default="datasets/oxford-iiit-pet")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--use_batch_norm", type=str2bool, default=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--classifier_checkpoint", type=str, default="checkpoints/classifier.pth")
    parser.add_argument("--localizer_checkpoint", type=str, default="checkpoints/localizer.pth")
    parser.add_argument("--unet_checkpoint", type=str, default="checkpoints/unet.pth")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["disabled", "offline", "online"])
    parser.add_argument("--wandb_project", type=str, default="da6401_assignment_2")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    return parser.parse_args()


def init_wandb(args: argparse.Namespace):
    if wandb is None or args.wandb_mode == "disabled":
        return None
    os.environ.setdefault("WANDB_DIR", resolve_path("wandb"))
    os.environ.setdefault("WANDB_CACHE_DIR", resolve_path("wandb/.cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", resolve_path("wandb/.config"))
    os.environ.setdefault("WANDB_DATA_DIR", resolve_path("wandb/.data"))
    try:
        kwargs = {
            "project": args.wandb_project,
            "config": vars(args),
            "mode": args.wandb_mode,
            "reinit": True,
        }
        if args.wandb_entity:
            kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name:
            kwargs["name"] = args.wandb_run_name
        return wandb.init(**kwargs)
    except Exception:
        return None


def build_model(args: argparse.Namespace, device: torch.device):
    if args.task == "classification":
        model = VGG11Classifier(
            num_classes=37,
            in_channels=3,
            dropout_p=args.dropout_p,
            use_batch_norm=args.use_batch_norm,
        )
        checkpoint_path = args.checkpoint or args.classifier_checkpoint
    elif args.task == "localization":
        model = VGG11Localizer(
            in_channels=3,
            dropout_p=args.dropout_p,
            image_size=args.image_size,
            use_batch_norm=args.use_batch_norm,
        )
        checkpoint_path = args.checkpoint or args.localizer_checkpoint
    elif args.task == "segmentation":
        model = VGG11UNet(
            num_classes=3,
            in_channels=3,
            dropout_p=max(args.dropout_p * 0.5, 0.1),
            use_batch_norm=args.use_batch_norm,
        )
        checkpoint_path = args.checkpoint or args.unet_checkpoint
    else:
        model = MultiTaskPerceptionModel(
            num_breeds=37,
            seg_classes=3,
            in_channels=3,
            classifier_path=args.classifier_checkpoint,
            localizer_path=args.localizer_checkpoint,
            unet_path=args.unet_checkpoint,
            dropout_p=args.dropout_p,
            use_batch_norm=args.use_batch_norm,
        )
        checkpoint_path = args.checkpoint

    if checkpoint_path:
        resolved = resolve_path(checkpoint_path)
        if os.path.exists(resolved):
            checkpoint = torch.load(resolved, map_location=device)
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def load_class_names(data_root: str) -> List[str]:
    resolved = resolve_path(data_root)
    if not os.path.isdir(resolved):
        return [f"class_{idx}" for idx in range(37)]
    dataset = OxfordIIITPetDataset(root=resolved, split="test", image_size=DEFAULT_IMAGE_SIZE, augment=False, download=False)
    return dataset.class_names


def preprocess_image(image_path: str, image_size: int):
    image = Image.open(image_path).convert("RGB")
    original = np.asarray(image)
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    image_np = np.asarray(resized).astype(np.float32) / 255.0
    image_np = (image_np - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    tensor = torch.from_numpy(np.transpose(image_np, (2, 0, 1))).float().unsqueeze(0)
    return tensor, original, np.asarray(resized)


def forward_task(model, task: str, images: torch.Tensor) -> Dict[str, torch.Tensor]:
    if task == "classification":
        return {"classification": model(images)}
    if task == "localization":
        return {"localization": model(images)}
    if task == "segmentation":
        return {"segmentation": model(images)}
    return model(images)


def evaluate_dataset(args: argparse.Namespace, model, device: torch.device):
    dataset = OxfordIIITPetDataset(
        root=resolve_path(args.data_root),
        split=args.split,
        image_size=args.image_size,
        augment=False,
        download=False,
    )
    dataloader = DataLoader(dataset, batch_size=args.num_samples, shuffle=False, num_workers=0)

    cls_preds: List[int] = []
    cls_targets: List[int] = []
    bbox_ious: List[float] = []
    dice_scores: List[float] = []
    pixel_scores: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            outputs = forward_task(model, args.task, images)

            if "classification" in outputs:
                logits = outputs["classification"]
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                cls_preds.extend(preds)
                cls_targets.extend(batch["label"].tolist())

            if "localization" in outputs:
                iou_values = box_iou_xywh(outputs["localization"].cpu(), batch["bbox"].float()).tolist()
                bbox_ious.extend(iou_values)

            if "segmentation" in outputs:
                dice_value = float(dice_score(outputs["segmentation"].cpu(), batch["mask"].long()).item())
                pixel_value = float(pixel_accuracy(outputs["segmentation"].cpu(), batch["mask"].long()).item())
                dice_scores.append(dice_value)
                pixel_scores.append(pixel_value)

    metrics = {}
    if cls_targets:
        metrics["classification_macro_f1"] = f1_score(cls_targets, cls_preds, average="macro", zero_division=0)
    if bbox_ious:
        metrics["localization_iou"] = float(np.mean(bbox_ious))
        metrics["localization_ap50_proxy"] = float(np.mean(np.asarray(bbox_ious) >= 0.5))
    if dice_scores:
        metrics["segmentation_dice"] = float(np.mean(dice_scores))
        metrics["segmentation_pixel_accuracy"] = float(np.mean(pixel_scores))
    return metrics


def save_feature_grid(feature_map: np.ndarray, output_path: str, title: str, max_channels: int = 16) -> None:
    channels = min(feature_map.shape[0], max_channels)
    cols = 4
    rows = int(math.ceil(channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for idx in range(rows * cols):
        axis = axes[idx // cols, idx % cols]
        axis.axis("off")
        if idx < channels:
            axis.imshow(feature_map[idx], cmap="viridis")
            axis.set_title(f"ch {idx}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_feature_maps(args: argparse.Namespace, model, device: torch.device) -> None:
    if not args.input:
        raise ValueError("--input is required for feature-map visualization.")

    first_activation = {}
    last_activation = {}

    def capture_first(_, __, output):
        first_activation["value"] = output.detach().cpu()[0].numpy()

    def capture_last(_, __, output):
        last_activation["value"] = output.detach().cpu()[0].numpy()

    hook_1 = model.encoder.block1.conv.register_forward_hook(capture_first)
    hook_2 = model.encoder.block5.conv2.register_forward_hook(capture_last)

    image_tensor, _, _ = preprocess_image(resolve_path(args.input), args.image_size)
    with torch.no_grad():
        _ = model(image_tensor.to(device))

    hook_1.remove()
    hook_2.remove()

    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    save_feature_grid(first_activation["value"], os.path.join(output_dir, "feature_maps_first_conv.png"), "First Convolutional Layer")
    save_feature_grid(last_activation["value"], os.path.join(output_dir, "feature_maps_last_conv.png"), "Last Convolutional Layer")


def run_activation_hist(args: argparse.Namespace, model, device: torch.device) -> None:
    if not args.input:
        raise ValueError("--input is required for activation-hist visualization.")

    captured = {}

    def capture(_, __, output):
        captured["value"] = output.detach().cpu().flatten().numpy()

    hook = model.encoder.block3.conv1.register_forward_hook(capture)
    image_tensor, _, _ = preprocess_image(resolve_path(args.input), args.image_size)
    with torch.no_grad():
        _ = model(image_tensor.to(device))
    hook.remove()

    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(captured["value"], bins=80, color="#2c7fb8")
    axis.set_title("Activation Distribution of the 3rd Convolutional Layer")
    axis.set_xlabel("Activation value")
    axis.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "third_conv_activation_hist.png"), dpi=160)
    plt.close(fig)


def run_single_image(args: argparse.Namespace, model, device: torch.device, class_names: Sequence[str]) -> None:
    if not args.input:
        raise ValueError("--input is required for image/showcase mode.")

    image_tensor, _, resized_image = preprocess_image(resolve_path(args.input), args.image_size)
    with torch.no_grad():
        outputs = forward_task(model, args.task, image_tensor.to(device))

    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    text_lines = []
    if "classification" in outputs:
        probs = torch.softmax(outputs["classification"], dim=1)[0].cpu()
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        text_lines.append(f"breed: {class_names[pred_idx]}")
        text_lines.append(f"conf: {confidence:.3f}")

    bbox_overlay = resized_image.copy()
    if "localization" in outputs:
        pred_box = outputs["localization"][0].cpu().tolist()
        bbox_overlay = draw_boxes(bbox_overlay, pred_box_xywh=pred_box, text_lines=text_lines or None)
        Image.fromarray(bbox_overlay).save(os.path.join(output_dir, "prediction_bbox.png"))

    if "segmentation" in outputs:
        pred_mask = torch.argmax(outputs["segmentation"], dim=1)[0].cpu().numpy()
        mask_overlay = blend_mask(resized_image, pred_mask)
        Image.fromarray(mask_overlay).save(os.path.join(output_dir, "prediction_mask.png"))

    if "localization" not in outputs and text_lines:
        Image.fromarray(draw_boxes(resized_image, text_lines=text_lines)).save(os.path.join(output_dir, "prediction_label.png"))


def run_bbox_table(args: argparse.Namespace, model, device: torch.device, run) -> None:
    dataset = OxfordIIITPetDataset(
        root=resolve_path(args.data_root),
        split=args.split,
        image_size=args.image_size,
        augment=False,
        download=False,
    )
    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    table = wandb.Table(columns=["image_id", "overlay", "confidence", "iou"]) if run is not None else None
    for index in range(min(args.num_samples, len(dataset))):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = forward_task(model, args.task, image)
        if "localization" not in outputs:
            raise ValueError("bbox_table mode requires a model with a localization head.")

        display_image = denormalize_image(sample["image"])
        pred_box = outputs["localization"][0].cpu()
        true_box = sample["bbox"].float()
        iou_value = float(box_iou_xywh(pred_box.unsqueeze(0), true_box.unsqueeze(0))[0].item())
        confidence = 1.0
        if "classification" in outputs:
            confidence = float(torch.softmax(outputs["classification"], dim=1).max().item())

        overlay = draw_boxes(
            display_image,
            pred_box_xywh=pred_box.tolist(),
            target_box_xywh=true_box.tolist(),
            text_lines=[f"conf: {confidence:.3f}", f"iou: {iou_value:.3f}"],
        )
        output_path = os.path.join(output_dir, f"bbox_{sample['image_id']}.png")
        Image.fromarray(overlay).save(output_path)
        if table is not None:
            table.add_data(sample["image_id"], wandb.Image(output_path), confidence, iou_value)

    if run is not None and table is not None:
        wandb.log({"bbox_predictions": table})


def run_mask_gallery(args: argparse.Namespace, model, device: torch.device, run) -> None:
    dataset = OxfordIIITPetDataset(
        root=resolve_path(args.data_root),
        split=args.split,
        image_size=args.image_size,
        augment=False,
        download=False,
    )
    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for index in range(min(args.num_samples, len(dataset))):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = forward_task(model, args.task, image)
        if "segmentation" not in outputs:
            raise ValueError("mask_gallery mode requires a model with a segmentation head.")
        pred_mask = torch.argmax(outputs["segmentation"], dim=1)[0].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(denormalize_image(sample["image"]))
        axes[0].set_title("Original")
        axes[1].imshow(sample["mask"].numpy(), cmap="viridis")
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred_mask, cmap="viridis")
        axes[2].set_title("Prediction")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        output_path = os.path.join(output_dir, f"mask_{sample['image_id']}.png")
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        if run is not None:
            wandb.log({f"mask_gallery/{sample['image_id']}": wandb.Image(output_path)})


def run_showcase(args: argparse.Namespace, model, device: torch.device, class_names: Sequence[str]) -> None:
    input_paths = []
    if args.input:
        input_paths.append(resolve_path(args.input))
    if args.input_dir:
        directory = resolve_path(args.input_dir)
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                input_paths.append(os.path.join(directory, name))
    if not input_paths:
        raise ValueError("Provide --input or --input_dir for showcase mode.")

    output_root = resolve_path(args.output_dir)
    os.makedirs(output_root, exist_ok=True)
    for image_path in input_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        args_for_image = argparse.Namespace(**vars(args))
        args_for_image.input = image_path
        args_for_image.output_dir = os.path.join(output_root, base_name)
        os.makedirs(args_for_image.output_dir, exist_ok=True)
        run_single_image(args_for_image, model, device, class_names)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    run = init_wandb(args)
    model = build_model(args, device)
    class_names = load_class_names(args.data_root)

    if args.mode == "dataset":
        metrics = evaluate_dataset(args, model, device)
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        if run is not None:
            wandb.log({f"eval/{name}": value for name, value in metrics.items()})
    elif args.mode == "feature_maps":
        run_feature_maps(args, model, device)
    elif args.mode == "activation_hist":
        run_activation_hist(args, model, device)
    elif args.mode == "bbox_table":
        run_bbox_table(args, model, device, run)
    elif args.mode == "mask_gallery":
        run_mask_gallery(args, model, device, run)
    elif args.mode == "showcase":
        run_showcase(args, model, device, class_names)
    else:
        run_single_image(args, model, device, class_names)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
