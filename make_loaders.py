# make_loaders.py
from datasets import load_dataset
from torch.utils.data import DataLoader
from hf_dataset import HFDropDataset

def get_loaders(target_classes, batch_size=32, img_size=224):
    ds = load_dataset("akshaybengani/hyper-kvasir", split="train")

    # podział bardzo prosty: 90% / 10%
    train_size = int(0.9 * len(ds))
    train_ds_raw = ds.select(range(train_size))
    val_ds_raw = ds.select(range(train_size, len(ds)))

    train_ds = HFDropDataset(train_ds_raw, target_classes, img_size, train=True)
    val_ds = HFDropDataset(val_ds_raw, target_classes, img_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader
