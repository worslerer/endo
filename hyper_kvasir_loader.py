import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

TARGET_CLASSES = [
    "ulcer", "polyp", "active_bleeding", "blood",
    "erythema", "erosion", "angiectasia", "IBD",
    "foreign_body", "esophagitis", "varices", "hematin",
    "celiac", "cancer", "lymphangioectasis", "other"
]

CLASS2IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}

class HyperKvasir17(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        metadata_path = os.path.join(root, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.samples = []

        for item in metadata:
            label = item["label"]
            if label not in CLASS2IDX:
                continue

            img_path = os.path.join(root, "images", item["image"])
            if os.path.exists(img_path):
                self.samples.append((img_path, CLASS2IDX[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)
