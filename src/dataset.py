import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

# Falcon label remapping (from your inspection)
LABEL_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    27: 4,
    39: 5,
}

class OffroadDataset(Dataset):
    def __init__(self, root, split="train", image_size=512):
        self.split = split
        self.image_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")

        self.images = sorted(os.listdir(self.image_dir))

        if split != "test":
            self.masks = sorted(os.listdir(self.mask_dir))

        if split == "train":
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.3),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
            ])

    def __len__(self):
        return len(self.images)

    def remap_mask(self, mask):
        remapped = np.zeros_like(mask)
        for src, dst in LABEL_MAP.items():
            remapped[mask == src] = dst
        return remapped

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.split != "test":
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = self.remap_mask(mask)

            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

            return image, mask
        else:
            image = self.transform(image=image)["image"]
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            return image, self.images[idx]
