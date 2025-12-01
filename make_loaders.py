# simple_loader.py

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

def get_loaders(root="kvasir", batch_size=32, img_size=224):

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # 1. Wczytujemy cały dataset jako jeden zbiór
    full_ds = ImageFolder(root, transform=transform_train)

    # 2. Robimy train / val split 90% / 10%
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size

    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # 3. Zmieniamy transform dla walidacji (optional)
    val_ds.dataset.transform = transform_val

    # 4. Tworzymy dataloadery
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Loaded dataset from {root}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Classes: {full_ds.classes}")

    return train_loader, val_loader
