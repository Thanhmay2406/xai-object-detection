"""Behavior-preserving COCO loader for torchvision detection models."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from xai_pruning.config import COCO_TO_MODEL_LABEL


class COCODetectionDataset(Dataset):
    """Load a COCO split while retaining empty images and original image IDs.

    The implementation follows Pipeline 03/04: image order comes directly from
    the JSON, invalid annotations are skipped only when width or height is not
    positive, and COCO ``area`` is retained when supplied. Training can enable
    the same horizontal flip used by the original Python dataset.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_path: str | Path,
        coco_to_model_label: Mapping[int, int] | None = None,
        *,
        train: bool = False,
        horizontal_flip_prob: float = 0.5,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.annotation_path = Path(annotation_path)
        self.annotation_file = str(self.annotation_path)
        self.coco_to_model_label = {
            int(key): int(value)
            for key, value in (coco_to_model_label or COCO_TO_MODEL_LABEL).items()
        }
        self.category_id_to_label = self.coco_to_model_label
        self.label_to_category_id = {
            value: key for key, value in self.coco_to_model_label.items()
        }
        self.train = bool(train)
        self.horizontal_flip_prob = float(horizontal_flip_prob)

        with self.annotation_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        self.coco = COCO(str(self.annotation_path))

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = {
            int(category["id"]): category["name"] for category in coco["categories"]
        }
        self.annotations_by_image: defaultdict[int, list[dict]] = defaultdict(list)
        for annotation in self.annotations:
            self.annotations_by_image[int(annotation["image_id"])].append(annotation)

        self.image_ids = [int(image["id"]) for image in self.images]
        self.image_info = {int(image["id"]): image for image in self.images}
        self.annotation_labels = sorted(
            {int(annotation["category_id"]) for annotation in self.annotations}
        )
        unknown = sorted(set(self.annotation_labels) - set(self.coco_to_model_label))
        if unknown:
            raise ValueError(f"Unmapped COCO category IDs: {unknown}")

        self.class_names = {0: "__background__"}
        for category_id, label in self.coco_to_model_label.items():
            if category_id in self.categories:
                self.class_names[label] = self.categories[category_id]
        self.num_classes = max(self.coco_to_model_label.values(), default=0) + 1

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_info = self.images[index]
        image_id = int(image_info["id"])
        image = Image.open(self.image_dir / image_info["file_name"]).convert("RGB")
        width, _ = image.size
        image = to_tensor(image)

        boxes = []
        model_labels = []
        coco_labels = []
        areas = []
        iscrowd = []

        for annotation in self.annotations_by_image[image_id]:
            x, y, box_width, box_height = annotation["bbox"]
            if box_width <= 0 or box_height <= 0:
                continue
            coco_label = int(annotation["category_id"])
            if coco_label not in self.coco_to_model_label:
                raise KeyError(f"No model label mapping for COCO category_id={coco_label}")
            boxes.append([x, y, x + box_width, y + box_height])
            coco_labels.append(coco_label)
            model_labels.append(self.coco_to_model_label[coco_label])
            areas.append(float(annotation.get("area", box_width * box_height)))
            iscrowd.append(int(annotation.get("iscrowd", 0)))

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if (
            self.train
            and self.horizontal_flip_prob > 0
            and random.random() < self.horizontal_flip_prob
        ):
            image = torch.flip(image, dims=[2])
            if boxes_tensor.numel() > 0:
                old_x1 = boxes_tensor[:, 0].clone()
                old_x2 = boxes_tensor[:, 2].clone()
                boxes_tensor[:, 0] = width - old_x2
                boxes_tensor[:, 2] = width - old_x1

        target = {
            "boxes": boxes_tensor,
            "labels": torch.tensor(model_labels, dtype=torch.int64),
            "coco_labels": torch.tensor(coco_labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
        return image, target


# Notebook-facing historical name retained as a compatibility alias.
COCODetectionDatasetMapped = COCODetectionDataset
COCOProbeDataset = COCODetectionDataset


def detection_collate_fn(batch):
    """Keep variable-sized detection samples as tuples."""

    return tuple(zip(*batch))


collate_fn = detection_collate_fn


def _build_loader(
    data_root: str | Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    train: bool,
    horizontal_flip_prob: float,
    coco_to_model_label: Mapping[int, int] | None,
    pin_memory: bool | None,
) -> DataLoader:
    split_dir = Path(data_root) / split
    dataset = COCODetectionDataset(
        split_dir,
        split_dir / "_annotations.coco.json",
        coco_to_model_label,
        train=train,
        horizontal_flip_prob=horizontal_flip_prob,
    )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=pin_memory,
    )


def build_train_loader(
    data_root: str | Path,
    batch_size: int = 4,
    num_workers: int = 2,
    *,
    horizontal_flip_prob: float = 0.5,
    coco_to_model_label: Mapping[int, int] | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Build the unchanged train split loader with horizontal flipping enabled."""

    return _build_loader(
        data_root,
        "train",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        train=True,
        horizontal_flip_prob=horizontal_flip_prob,
        coco_to_model_label=coco_to_model_label,
        pin_memory=pin_memory,
    )


def build_valid_loader(
    data_root: str | Path,
    batch_size: int = 2,
    num_workers: int = 2,
    *,
    coco_to_model_label: Mapping[int, int] | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Build the deterministic validation loader."""

    return _build_loader(
        data_root,
        "valid",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        train=False,
        horizontal_flip_prob=0.0,
        coco_to_model_label=coco_to_model_label,
        pin_memory=pin_memory,
    )


def build_test_loader(
    data_root: str | Path,
    batch_size: int = 2,
    num_workers: int = 2,
    *,
    coco_to_model_label: Mapping[int, int] | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Build the deterministic test loader without filtering empty images."""

    return _build_loader(
        data_root,
        "test",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        train=False,
        horizontal_flip_prob=0.0,
        coco_to_model_label=coco_to_model_label,
        pin_memory=pin_memory,
    )
