
from model_creator import ModelWrapper
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics
import datetime
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model_wrapper, device="cuda"):
        self.model_wrapper = model_wrapper
        self.device = device

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"{model_wrapper.model_name}_at_{timestamp}"
        self.writer = SummaryWriter(f"runs/experiment_{experiment_name}")

    def train_one_epoch(self, model, loader, optimizer, loss_fn, epoch_index):

        running_loss = 0.
        last_loss = 0.
        model.train()

        for i, data in enumerate(loader):

            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss_value = loss_fn(outputs, labels)
            loss_value.backward()
            optimizer.step()

            running_loss += loss_value.item()
            if i % 100 == 0:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(loader) + i + 1
                self.writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def validate_one_epoch(self, model, loader, loss_fn):

        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(self.device)
                vlabels = vlabels.to(self.device)

                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()

        return running_vloss / len(loader)

    def train_one_fold(self, train_loader, val_loader, epochs, fold_id):

        best_val_loss = float("inf")
        model, loss_fn, activation_fn, optimizer = self.model_wrapper.build_model()
        model.to(self.device)
        for epoch in range(epochs):

            print('EPOCH {}:'.format(epoch + 1))
            train_loss = self.train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, epoch_index=epoch)
            val_loss = self.validate_one_epoch(model=model, loader=val_loader, loss_fn=loss_fn)

            print(f"Train Loss: {train_loss}")
            print(f"Validation Loss: {val_loss}")

            self.writer.add_scalars(
                f"Fold{fold_id}_Loss",
                {"train": train_loss, "val": val_loss},
                epoch
            )

            self.writer.flush()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_model_fold_{fold_id}.pth")
                print("model saved")

        return best_val_loss