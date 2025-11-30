# run_train.py
import torch
from wrapper import ModelWrapper
from trainer import Trainer
from make_loaders import get_loaders

device = "cuda"

TARGET_CLASSES = ["ulcer", "polyp", "blood"]

train_loader, val_loader = get_loaders(
    target_classes=TARGET_CLASSES,
    batch_size=32,
    img_size=224
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
