import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model_creator import ModelWrapper
from trainer import Trainer

# ------------------------------
# 1. Dataset do testów (CIFAR10)
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Żeby pasowało do ResNet / MobileNet
    transforms.ToTensor(),
])

train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
val_ds   = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

# ------------------------------
# 2. Stwórz ModelWrapper
# ------------------------------
model_wrapper = ModelWrapper(
    model_name="resnet50",
    num_classes=10,            # CIFAR10 ma 10 klas
    learning_rate=0.001,
    weight_decay=0.0001,
    multi_label=False,
    opt_name="AdamW",
    focal_loss=False          # CIFAR10 = klasy wzajemnie wykluczające
)

# ------------------------------
# 3. Stwórz Trainer
# ------------------------------
trainer = Trainer(model_wrapper=model_wrapper, device="cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 4. Odpal trening jednego folda
# ------------------------------
trainer.train_one_fold(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=3,
    fold_id=1
)
