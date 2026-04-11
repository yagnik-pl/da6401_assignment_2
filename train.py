"""Training entrypoint for all Assignment 2 tasks."""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from common import box_iou_xywh, dice_loss, dice_score, pixel_accuracy, resolve_path, save_checkpoint, set_seed
from data.pets_dataset import OxfordIIITPetDataset, download_oxford_pet
from losses import IoULoss
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
    parser = argparse.ArgumentParser(description="Train Assignment 2 models on Oxford-IIIT Pet")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "localization", "segmentation", "multitask"])
    parser.add_argument("--data_root", type=str, default="datasets/oxford-iiit-pet")
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--prepare_data_only", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_batch_norm", type=str2bool, default=True)
    parser.add_argument("--freeze_strategy", type=str, default="none", choices=["none", "strict", "partial"])
    parser.add_argument("--pretrained", type=str2bool, default=False,
                        help="Bootstrap encoder from torchvision VGG11-BN ImageNet weights before training.")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of linear LR warm-up epochs before cosine decay kicks in.")
    parser.add_argument("--load_encoder_from_classifier", type=str2bool, default=True)
    parser.add_argument("--classifier_checkpoint", type=str, default="checkpoints/classifier.pth")
    parser.add_argument("--localizer_checkpoint", type=str, default="checkpoints/localizer.pth")
    parser.add_argument("--unet_checkpoint", type=str, default="checkpoints/unet.pth")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classification_loss_weight", type=float, default=1.0)
    parser.add_argument("--bbox_mse_weight", type=float, default=1.0)
    parser.add_argument("--bbox_iou_weight", type=float, default=1.0)
    parser.add_argument("--seg_ce_weight", type=float, default=1.0)
    parser.add_argument("--seg_dice_weight", type=float, default=1.0)
    parser.add_argument("--evaluate_test", action="store_true")
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


def build_dataloaders(args: argparse.Namespace, device: torch.device):
    train_dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="train",
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        augment=True,
        download=args.download_data,
    )
    val_dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="val",
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        augment=False,
        download=False,
    )
    test_dataset = OxfordIIITPetDataset(
        root=args.data_root,
        split="test",
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        augment=False,
        download=False,
    )
    if len(test_dataset) == 0:
        test_dataset = OxfordIIITPetDataset(
            root=args.data_root,
            split="val",
            image_size=args.image_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
            augment=False,
            download=False,
        )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def _load_pretrained_encoder(model: nn.Module) -> None:
    """Attempt to load ImageNet pretrained weights into any VGG11Encoder found in *model*."""
    # Direct encoder (classification / localization / segmentation tasks)
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "load_pretrained_weights"):
        ok = encoder.load_pretrained_weights()
        if ok:
            print("  [pretrained] Loaded ImageNet VGG11-BN weights into encoder.")
        return
    # MultiTaskPerceptionModel: initialise each sub-model's encoder separately so that
    # all three tasks start from the same strong feature extractor.
    for attr in ("classifier_model", "localizer_model", "segmenter_model"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        enc = getattr(sub, "encoder", None)
        if enc is not None and hasattr(enc, "load_pretrained_weights"):
            ok = enc.load_pretrained_weights()
            if ok:
                print(f"  [pretrained] Loaded ImageNet VGG11-BN weights into {attr}.encoder.")


def freeze_encoder_layers(encoder: nn.Module, strategy: str) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad = True

    if strategy == "strict":
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        return

    if strategy == "partial":
        for block_name in ("block1", "block2", "block3"):
            for parameter in getattr(encoder, block_name).parameters():
                parameter.requires_grad = False


def load_classifier_encoder_weights(model: nn.Module, checkpoint_path: str, args: argparse.Namespace) -> None:
    resolved = resolve_path(checkpoint_path)
    if not os.path.exists(resolved) or not hasattr(model, "encoder"):
        return
    pretrained = VGG11Classifier(
        num_classes=37,
        in_channels=3,
        dropout_p=args.dropout_p,
        use_batch_norm=args.use_batch_norm,
    )
    checkpoint = torch.load(resolved, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    pretrained.load_state_dict(state_dict, strict=False)
    model.encoder.load_state_dict(pretrained.encoder.state_dict(), strict=False)


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.task == "classification":
        model = VGG11Classifier(
            num_classes=37,
            in_channels=3,
            dropout_p=args.dropout_p,
            use_batch_norm=args.use_batch_norm,
        )
    elif args.task == "localization":
        model = VGG11Localizer(
            in_channels=3,
            dropout_p=args.dropout_p,
            image_size=args.image_size,
            use_batch_norm=args.use_batch_norm,
        )
        if args.load_encoder_from_classifier:
            load_classifier_encoder_weights(model, args.classifier_checkpoint, args)
    elif args.task == "segmentation":
        model = VGG11UNet(
            num_classes=3,
            in_channels=3,
            dropout_p=max(args.dropout_p * 0.5, 0.1),
            use_batch_norm=args.use_batch_norm,
        )
        if args.load_encoder_from_classifier:
            load_classifier_encoder_weights(model, args.classifier_checkpoint, args)
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

    if getattr(args, "pretrained", False):
        _load_pretrained_encoder(model)

    if hasattr(model, "encoder"):
        freeze_encoder_layers(model.encoder, args.freeze_strategy)
    return model.to(device)


def build_optimizer(args: argparse.Namespace, model: nn.Module):
    # When using a pretrained encoder, use a 10x smaller LR for the backbone so that
    # fine-tuned features are not destroyed by the larger head learning rate.
    if getattr(args, "pretrained", False) and hasattr(model, "encoder"):
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
        param_groups = [
            {"params": encoder_params, "lr": args.learning_rate * 0.1},
            {"params": head_params,    "lr": args.learning_rate},
        ]
        if args.optimizer == "sgd":
            return torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.optimizer == "adamw":
            return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        return torch.optim.Adam(param_groups, weight_decay=args.weight_decay)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    return torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)


def forward_task(model: nn.Module, task: str, images: torch.Tensor) -> Dict[str, torch.Tensor]:
    if task == "classification":
        return {"classification": model(images)}
    if task == "localization":
        return {"localization": model(images)}
    if task == "segmentation":
        return {"segmentation": model(images)}
    return model(images)


def compute_losses_and_stats(
    args: argparse.Namespace,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, object],
    device: torch.device,
    cls_criterion: nn.Module,
    mse_criterion: nn.Module,
    iou_criterion: nn.Module,
    seg_criterion: nn.Module,
):
    loss = torch.zeros((), device=device)
    stats = {}

    if "classification" in outputs:
        labels = batch["label"].to(device)
        logits = outputs["classification"]
        cls_loss = cls_criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)
        loss = loss + args.classification_loss_weight * cls_loss
        stats.update(
            {
                "cls_loss": float(cls_loss.detach().item()),
                "cls_preds": predictions.detach().cpu(),
                "cls_targets": labels.detach().cpu(),
                "cls_correct": int((predictions == labels).sum().item()),
            }
        )

    if "localization" in outputs:
        target_boxes = batch["bbox"].to(device)
        pred_boxes = outputs["localization"]
        scale = float(args.image_size)
        mse_value = mse_criterion(pred_boxes / scale, target_boxes / scale)
        iou_loss_value = iou_criterion(pred_boxes, target_boxes)
        iou_value = box_iou_xywh(pred_boxes.detach(), target_boxes.detach()).mean()
        loss = loss + args.bbox_mse_weight * mse_value + args.bbox_iou_weight * iou_loss_value
        stats.update(
            {
                "bbox_mse": float(mse_value.detach().item()),
                "bbox_iou_loss": float(iou_loss_value.detach().item()),
                "bbox_iou": float(iou_value.detach().item()),
            }
        )

    if "segmentation" in outputs:
        masks = batch["mask"].to(device)
        logits = outputs["segmentation"]
        seg_ce = seg_criterion(logits, masks)
        seg_dice_loss = dice_loss(logits, masks)
        seg_dice_score = dice_score(logits.detach(), masks.detach())
        seg_pixel_acc = pixel_accuracy(logits.detach(), masks.detach())
        loss = loss + args.seg_ce_weight * seg_ce + args.seg_dice_weight * seg_dice_loss
        stats.update(
            {
                "seg_ce": float(seg_ce.detach().item()),
                "seg_dice_loss": float(seg_dice_loss.detach().item()),
                "seg_dice": float(seg_dice_score.detach().item()),
                "seg_pixel_acc": float(seg_pixel_acc.detach().item()),
            }
        )

    return loss, stats


def init_epoch_meter():
    return {
        "num_samples": 0,
        "loss_sum": 0.0,
        "cls_loss_sum": 0.0,
        "bbox_mse_sum": 0.0,
        "bbox_iou_loss_sum": 0.0,
        "bbox_iou_sum": 0.0,
        "seg_ce_sum": 0.0,
        "seg_dice_loss_sum": 0.0,
        "seg_dice_sum": 0.0,
        "seg_pixel_acc_sum": 0.0,
        "cls_correct": 0,
        "cls_preds": [],
        "cls_targets": [],
    }


def summarize_epoch(meter, task: str):
    count = max(meter["num_samples"], 1)
    summary = {"loss": meter["loss_sum"] / count}

    if meter["cls_targets"]:
        summary["classification_accuracy"] = meter["cls_correct"] / count
        summary["classification_macro_f1"] = f1_score(
            meter["cls_targets"],
            meter["cls_preds"],
            average="macro",
            zero_division=0,
        )
        summary["classification_loss"] = meter["cls_loss_sum"] / count

    if meter["bbox_iou_sum"] > 0 or task in {"localization", "multitask"}:
        summary["localization_mse"] = meter["bbox_mse_sum"] / count
        summary["localization_iou_loss"] = meter["bbox_iou_loss_sum"] / count
        summary["localization_iou"] = meter["bbox_iou_sum"] / count

    if meter["seg_ce_sum"] > 0 or task in {"segmentation", "multitask"}:
        summary["segmentation_ce"] = meter["seg_ce_sum"] / count
        summary["segmentation_dice_loss"] = meter["seg_dice_loss_sum"] / count
        summary["segmentation_dice"] = meter["seg_dice_sum"] / count
        summary["segmentation_pixel_accuracy"] = meter["seg_pixel_acc_sum"] / count

    return summary


def run_epoch(
    args: argparse.Namespace,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)

    cls_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    mse_criterion = nn.MSELoss()
    iou_criterion = IoULoss(reduction="mean")
    seg_criterion = nn.CrossEntropyLoss()
    meter = init_epoch_meter()

    for batch in dataloader:
        images = batch["image"].to(device)
        outputs = forward_task(model, args.task, images)
        loss, stats = compute_losses_and_stats(
            args=args,
            outputs=outputs,
            batch=batch,
            device=device,
            cls_criterion=cls_criterion,
            mse_criterion=mse_criterion,
            iou_criterion=iou_criterion,
            seg_criterion=seg_criterion,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_size = images.size(0)
        meter["num_samples"] += batch_size
        meter["loss_sum"] += float(loss.detach().item()) * batch_size

        if "cls_loss" in stats:
            meter["cls_loss_sum"] += stats["cls_loss"] * batch_size
            meter["cls_correct"] += stats["cls_correct"]
            meter["cls_preds"].extend(stats["cls_preds"].tolist())
            meter["cls_targets"].extend(stats["cls_targets"].tolist())

        if "bbox_mse" in stats:
            meter["bbox_mse_sum"] += stats["bbox_mse"] * batch_size
            meter["bbox_iou_loss_sum"] += stats["bbox_iou_loss"] * batch_size
            meter["bbox_iou_sum"] += stats["bbox_iou"] * batch_size

        if "seg_ce" in stats:
            meter["seg_ce_sum"] += stats["seg_ce"] * batch_size
            meter["seg_dice_loss_sum"] += stats["seg_dice_loss"] * batch_size
            meter["seg_dice_sum"] += stats["seg_dice"] * batch_size
            meter["seg_pixel_acc_sum"] += stats["seg_pixel_acc"] * batch_size

    return summarize_epoch(meter, args.task)


def score_from_metrics(task: str, metrics: Dict[str, float]) -> float:
    if task == "classification":
        return metrics.get("classification_macro_f1", 0.0)
    if task == "localization":
        return metrics.get("localization_iou", 0.0)
    if task == "segmentation":
        return metrics.get("segmentation_dice", 0.0)
    components = [
        metrics.get("classification_macro_f1", 0.0),
        metrics.get("localization_iou", 0.0),
        metrics.get("segmentation_dice", 0.0),
    ]
    return sum(components) / len(components)


def format_metrics(metrics: Dict[str, float]) -> str:
    ordered = [f"{name}={value:.4f}" for name, value in metrics.items()]
    return ", ".join(ordered)


def default_checkpoint_name(task: str) -> str:
    mapping = {
        "classification": "classifier.pth",
        "localization": "localizer.pth",
        "segmentation": "unet.pth",
        "multitask": "multitask.pth",
    }
    return mapping[task]


def maybe_resume(model: nn.Module, optimizer, resume_path: str, device: torch.device, learning_rate: float):
    if not resume_path:
        return 0, float("-inf")
    resolved = resolve_path(resume_path)
    if not os.path.exists(resolved):
        return 0, float("-inf")

    checkpoint = torch.load(resolved, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    if isinstance(checkpoint, dict) and "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate
        except (ValueError, KeyError):
            # Optimizer param-group mismatch (e.g. pretrained vs non-pretrained run).
            # Safe to ignore — LR scheduler will set the LR correctly.
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate
    start_epoch = int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0
    best_metric = float(checkpoint.get("best_metric", float("-inf"))) if isinstance(checkpoint, dict) else float("-inf")
    return start_epoch, best_metric


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data_root = resolve_path(args.data_root)
    args.data_root = data_root

    if args.download_data:
        download_oxford_pet(args.data_root)
    if args.prepare_data_only:
        print(f"Dataset prepared at: {args.data_root}")
        return

    run = init_wandb(args)
    train_loader, val_loader, test_loader = build_dataloaders(args, device)
    model = build_model(args, device)
    optimizer = build_optimizer(args, model)

    start_epoch, best_score = maybe_resume(model, optimizer, args.resume, device, args.learning_rate)
    checkpoint_path = resolve_path(os.path.join(args.checkpoint_dir, default_checkpoint_name(args.task)))

    remaining_epochs = max(args.epochs - start_epoch, 1)
    warmup_epochs = min(getattr(args, "warmup_epochs", 5), remaining_epochs)
    cosine_epochs = max(remaining_epochs - warmup_epochs, 1)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=args.learning_rate * 1e-3
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(args, model, train_loader, device, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(args, model, val_loader, device, optimizer=None)

        current_score = score_from_metrics(args.task, val_metrics)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  train: {format_metrics(train_metrics)}")
        print(f"  val:   {format_metrics(val_metrics)}")

        if run is not None:
            log_payload = {"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"]}
            log_payload.update({f"train/{key}": value for key, value in train_metrics.items()})
            log_payload.update({f"val/{key}": value for key, value in val_metrics.items()})
            wandb.log(log_payload)

        if current_score > best_score:
            best_score = current_score
            save_checkpoint(
                checkpoint_path,
                model,
                epoch=epoch + 1,
                best_metric=best_score,
                extra={"optimizer_state": optimizer.state_dict(), "config": vars(args)},
            )
            print(f"  saved best checkpoint to {checkpoint_path}")

    if args.evaluate_test and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
        with torch.no_grad():
            test_metrics = run_epoch(args, model, test_loader, device, optimizer=None)
        print(f"Test metrics: {format_metrics(test_metrics)}")
        if run is not None:
            wandb.log({f"test/{key}": value for key, value in test_metrics.items()})

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()