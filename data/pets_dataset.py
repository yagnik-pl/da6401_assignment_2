"""Oxford-IIIT Pet multi-task dataset loader."""

from __future__ import annotations

import os
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from common import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

try:
    import albumentations as A
except ImportError:
    A = None


IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"


def _download_file(url: str, destination: str) -> None:
    parent = os.path.dirname(destination)
    if parent:
        os.makedirs(parent, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def download_oxford_pet(root: str) -> None:
    """Download and extract the Oxford-IIIT Pet dataset if missing."""
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    images_dir = os.path.join(root, "images")
    annotations_dir = os.path.join(root, "annotations")
    if os.path.isdir(images_dir) and os.path.isdir(annotations_dir):
        return

    archives = [
        (IMAGES_URL, os.path.join(root, "images.tar.gz")),
        (ANNOTATIONS_URL, os.path.join(root, "annotations.tar.gz")),
    ]
    for url, archive_path in archives:
        if not os.path.exists(archive_path):
            _download_file(url, archive_path)
        with tarfile.open(archive_path, "r:gz") as tar_file:
            tar_file.extractall(root)


def _breed_name_from_image_id(image_id: str) -> str:
    return " ".join(image_id.split("_")[:-1]).replace("-", " ")


def _parse_split_file(split_path: str) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    with open(split_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            image_id = parts[0]
            class_index = int(parts[1]) - 1
            samples.append((image_id, class_index))
    return samples


def _parse_bbox(xml_path: str) -> Tuple[float, float, float, float]:
    root = ET.parse(xml_path).getroot()
    bbox = root.find(".//bndbox")
    if bbox is None:
        raise ValueError(f"Could not locate bounding box in '{xml_path}'.")
    xmin = float(bbox.findtext("xmin", default="0"))
    ymin = float(bbox.findtext("ymin", default="0"))
    xmax = float(bbox.findtext("xmax", default="0"))
    ymax = float(bbox.findtext("ymax", default="0"))
    return xmin, ymin, xmax, ymax


def _has_required_files(root: str, image_id: str) -> bool:
    image_path = os.path.join(root, "images", f"{image_id}.jpg")
    trimap_path = os.path.join(root, "annotations", "trimaps", f"{image_id}.png")
    xml_path = os.path.join(root, "annotations", "xmls", f"{image_id}.xml")
    return os.path.exists(image_path) and os.path.exists(trimap_path) and os.path.exists(xml_path)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    image = (image - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    return image


def _remap_trimap(mask: np.ndarray) -> np.ndarray:
    """Map Oxford trimap labels to background-first indices.

    Original labels:
    1 -> pet
    2 -> background
    3 -> border

    Remapped labels used here:
    0 -> background
    1 -> pet
    2 -> border
    """
    remapped = np.zeros_like(mask, dtype=np.int64)
    remapped[mask == 2] = 0
    remapped[mask == 1] = 1
    remapped[mask == 3] = 2
    return remapped


def _expand_xyxy_box(xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int, scale: float):
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    box_w = max(xmax - xmin, 1.0) * scale
    box_h = max(ymax - ymin, 1.0) * scale
    left = max(0.0, center_x - 0.5 * box_w)
    top = max(0.0, center_y - 0.5 * box_h)
    right = min(float(width), center_x + 0.5 * box_w)
    bottom = min(float(height), center_y + 0.5 * box_h)
    return left, top, right, bottom


class _FallbackTransform:
    """Lightweight transform path when albumentations is unavailable."""

    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE, augment: bool = False):
        self.image_size = image_size
        self.augment = augment

    def __call__(self, image: np.ndarray, mask: np.ndarray, bbox: Sequence[float]):
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L")
        orig_w, orig_h = image_pil.size

        xmin, ymin, xmax, ymax = bbox
        if self.augment and np.random.rand() < 0.5:
            image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
            mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
            xmin, xmax = orig_w - xmax, orig_w - xmin

        if self.augment and np.random.rand() < 0.3:
            image_pil = ImageEnhance.Brightness(image_pil).enhance(0.8 + 0.4 * np.random.rand())

        image_pil = image_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)

        scale_x = self.image_size / float(orig_w)
        scale_y = self.image_size / float(orig_h)
        bbox = [
            xmin * scale_x,
            ymin * scale_y,
            xmax * scale_x,
            ymax * scale_y,
        ]

        image_np = _normalize_image(np.asarray(image_pil))
        mask_np = np.asarray(mask_pil, dtype=np.int64)
        return image_np, mask_np, bbox


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet dataset that returns labels, bounding boxes, and trimaps."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = DEFAULT_IMAGE_SIZE,
        val_ratio: float = 0.1,
        seed: int = 42,
        augment: bool = False,
        download: bool = False,
        crop_to_bbox: bool = False,
        crop_scale: float = 2.2,
    ):
        self.root = os.path.abspath(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.val_ratio = val_ratio
        self.seed = seed
        self.crop_to_bbox = crop_to_bbox
        self.crop_scale = crop_scale

        if download:
            download_oxford_pet(self.root)

        annotations_dir = os.path.join(self.root, "annotations")
        images_dir = os.path.join(self.root, "images")
        if not os.path.isdir(annotations_dir) or not os.path.isdir(images_dir):
            raise FileNotFoundError(
                "Oxford-IIIT Pet dataset not found. Download it with download=True "
                "or place it under '<root>/images' and '<root>/annotations'."
            )

        self.images_dir = images_dir
        self.trimaps_dir = os.path.join(annotations_dir, "trimaps")
        self.xml_dir = os.path.join(annotations_dir, "xmls")

        self.class_names = self._build_class_names(annotations_dir)
        self.samples = self._build_samples(annotations_dir)
        self.transform = self._build_transform()

    def _build_class_names(self, annotations_dir: str) -> List[str]:
        class_names = [""] * 37
        for split_name in ("trainval.txt", "test.txt"):
            split_path = os.path.join(annotations_dir, split_name)
            if not os.path.exists(split_path):
                continue
            for image_id, class_index in _parse_split_file(split_path):
                class_names[class_index] = _breed_name_from_image_id(image_id)
        return [name if name else f"class_{idx}" for idx, name in enumerate(class_names)]

    def _build_samples(self, annotations_dir: str) -> List[Tuple[str, int]]:
        if self.split == "test":
            samples = _parse_split_file(os.path.join(annotations_dir, "test.txt"))
            return [(image_id, label) for image_id, label in samples if _has_required_files(self.root, image_id)]

        trainval_samples = _parse_split_file(os.path.join(annotations_dir, "trainval.txt"))
        trainval_samples = [
            (image_id, label) for image_id, label in trainval_samples if _has_required_files(self.root, image_id)
        ]
        image_ids = [image_id for image_id, _ in trainval_samples]
        labels = [label for _, label in trainval_samples]
        train_ids, val_ids = train_test_split(
            image_ids,
            test_size=self.val_ratio,
            random_state=self.seed,
            shuffle=True,
            stratify=labels,
        )
        selected_ids = set(train_ids if self.split == "train" else val_ids)
        return [(image_id, label) for image_id, label in trainval_samples if image_id in selected_ids]

    def _build_transform(self):
        if A is None:
            return _FallbackTransform(image_size=self.image_size, augment=self.augment)

        transforms = []
        if self.augment:
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.15, rotate_limit=15, border_mode=0, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                ]
            )
        transforms.extend(
            [
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"], min_visibility=0.0),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_id, label = self.samples[index]
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        trimap_path = os.path.join(self.trimaps_dir, f"{image_id}.png")
        xml_path = os.path.join(self.xml_dir, f"{image_id}.xml")

        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = _remap_trimap(np.asarray(Image.open(trimap_path), dtype=np.int64))
        bbox_xyxy = _parse_bbox(xml_path)
        original_size = np.asarray(image.shape[:2], dtype=np.int64)

        if self.crop_to_bbox:
            orig_h, orig_w = image.shape[:2]
            xmin, ymin, xmax, ymax = bbox_xyxy
            crop_left, crop_top, crop_right, crop_bottom = _expand_xyxy_box(
                xmin,
                ymin,
                xmax,
                ymax,
                width=orig_w,
                height=orig_h,
                scale=self.crop_scale,
            )
            left_i = int(np.floor(crop_left))
            top_i = int(np.floor(crop_top))
            right_i = int(np.ceil(crop_right))
            bottom_i = int(np.ceil(crop_bottom))

            image = image[top_i:bottom_i, left_i:right_i]
            mask = mask[top_i:bottom_i, left_i:right_i]
            bbox_xyxy = (
                xmin - left_i,
                ymin - top_i,
                xmax - left_i,
                ymax - top_i,
            )
            original_size = np.asarray(image.shape[:2], dtype=np.int64)

        if A is None:
            image, mask, bbox_xyxy = self.transform(image=image, mask=mask, bbox=bbox_xyxy)
        else:
            transformed = self.transform(
                image=image,
                mask=mask,
                bboxes=[bbox_xyxy],
                bbox_labels=[1],
            )
            image = transformed["image"]
            mask = transformed["mask"]
            if transformed["bboxes"]:
                bbox_xyxy = transformed["bboxes"][0]
            # else bbox_xyxy keeps its pre-transform value (already in resized space via Resize)

        xmin, ymin, xmax, ymax = bbox_xyxy
        bbox_xywh = np.asarray(
            [
                0.5 * (xmin + xmax),
                0.5 * (ymin + ymax),
                max(xmax - xmin, 1.0),
                max(ymax - ymin, 1.0),
            ],
            dtype=np.float32,
        )

        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        bbox_tensor = torch.from_numpy(bbox_xywh)
        bbox_xyxy_tensor = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "bbox": bbox_tensor,
            "bbox_xyxy": bbox_xyxy_tensor,
            "mask": mask_tensor,
            "image_id": image_id,
            "breed_name": self.class_names[label],
            "original_size": torch.from_numpy(original_size),
        }