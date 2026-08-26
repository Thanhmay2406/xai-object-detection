import random
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class COCODetectionDataset(Dataset):
    """
    COCO-format dataset for torchvision Faster R-CNN.

    Output
    ------
    image:
        FloatTensor [C, H, W], range [0, 1]

    target:
        {
            "boxes": FloatTensor[N, 4],   # XYXY
            "labels": Int64Tensor[N],
            "image_id": Int64Tensor[1],
            "area": FloatTensor[N],
            "iscrowd": Int64Tensor[N],
        }
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        train: bool = False,
        horizontal_flip_prob: float = 0.5,
        category_id_to_label: Optional[Dict[int, int]] = None,
    ):
        self.image_dir = Path(image_dir)
        self.annotation_file = annotation_file
        self.train = train
        self.horizontal_flip_prob = horizontal_flip_prob

        self.coco = COCO(annotation_file)

        # Image IDs
        self.image_ids = sorted(self.coco.getImgIds())

        # COCO category IDs can be non-contiguous.
        category_ids = sorted(self.coco.getCatIds())

        if category_id_to_label is None:
            # 0 reserved for background.
            self.category_id_to_label = {
                category_id: label
                for label, category_id in enumerate(category_ids, start=1)
            }
        else:
            self.category_id_to_label = category_id_to_label

        self.label_to_category_id = {
            label: category_id
            for category_id, label in self.category_id_to_label.items()
        }

        self.num_classes = len(self.category_id_to_label) + 1

        categories = self.coco.loadCats(category_ids)

        self.class_names = {
            0: "__background__"
        }

        for category in categories:
            category_id = category["id"]

            if category_id in self.category_id_to_label:
                label = self.category_id_to_label[category_id]
                self.class_names[label] = category["name"]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_info = self.coco.loadImgs(image_id)[0]

        image_path = self.image_dir / image_info["file_name"]

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        annotation_ids = self.coco.getAnnIds(
            imgIds=[image_id],
            iscrowd=None,
        )

        annotations = self.coco.loadAnns(annotation_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            category_id = ann["category_id"]

            if category_id not in self.category_id_to_label:
                continue

            # COCO bbox: [x, y, width, height]
            x, y, w, h = ann["bbox"]

            if w <= 0 or h <= 0:
                continue

            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))

            x2 = min(float(width), float(x + w))
            y2 = min(float(height), float(y + h))

            # Remove degenerate boxes.
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])

            labels.append(
                self.category_id_to_label[category_id]
            )

            areas.append((x2 - x1) * (y2 - y1))

            iscrowd.append(
                int(ann.get("iscrowd", 0))
            )

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        else:
            boxes = torch.tensor(
                boxes,
                dtype=torch.float32,
            )

            labels = torch.tensor(
                labels,
                dtype=torch.int64,
            )

            areas = torch.tensor(
                areas,
                dtype=torch.float32,
            )

            iscrowd = torch.tensor(
                iscrowd,
                dtype=torch.int64,
            )

        # PIL -> uint8 tensor -> float [0,1]
        image = pil_to_tensor(image).float() / 255.0

        # Simple baseline augmentation.
        if (
            self.train
            and self.horizontal_flip_prob > 0
            and random.random() < self.horizontal_flip_prob
        ):
            image = torch.flip(image, dims=[2])

            if boxes.numel() > 0:
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()

                boxes[:, 0] = width - old_x2
                boxes[:, 2] = width - old_x1

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(
                [image_id],
                dtype=torch.int64,
            ),
            "area": areas,
            "iscrowd": iscrowd,
        }

        return image, target


def collate_fn(batch):
    """
    Detection models receive List[Tensor], not one stacked tensor.
    """
    return tuple(zip(*batch))
