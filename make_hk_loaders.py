from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from hyper_kvasir_loader import HyperKvasir17

def get_hk_loaders(root="hyperkvasir/pathology", batch_size=32):

    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    full_ds = HyperKvasir17(root=root, transform=transform_train)

    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size

    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    val_ds.dataset.transform = transform_val

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)

    print(f"Loaded {len(full_ds)} pathology images.")
    return train_loader, val_loader
