# run_train.py
import torch
from model_creator import ModelWrapper
from trainer import Trainer
#from make_loaders import get_loaders
#from loaders import get_loaders
from make_hk_loaders import get_hk_loaders

device = "cuda"

TARGET_CLASSES = ["ulcer", "polyp", "blood"]
train_loader, val_loader = get_hk_loaders(root="hyperkvasir/pathology", batch_size=32)

train_loader, val_loader = get_hk_loaders(
    root="hyperkvasir/pathology",
    batch_size=32
)

MODELS = ["resnet50", "mobilenetv2", "densenet121"]

trainer = Trainer(device=device)

for model_name in MODELS:
    print(f"\n--- Training {model_name} ---\n")

    wrapper = ModelWrapper(
        model_name=model_name,
        num_classes=len(TARGET_CLASSES),
        learning_rate=0.0001,
        weight_decay=1e-4,
        multi_label=False  # tu jest multiclass
    )

    trainer.train_one_fold(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        fold_id=model_name
    )
