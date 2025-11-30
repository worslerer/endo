# hf_dataset.py
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class HFDropDataset(Dataset):
    def __init__(self, hf_dataset, target_classes, img_size=224, train=True):
        self.ds = [
            x for x in hf_dataset if x["label"] in target_classes
        ]
        self.target_classes = target_classes
        self.class2id = {c: i for i, c in enumerate(target_classes)}

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"]
        label_name = sample["label"]

        img = self.transform(img)

        label = torch.tensor(self.class2id[label_name], dtype=torch.long)

        return img, label
