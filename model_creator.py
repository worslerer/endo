import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
from torchvision.models import resnet50, mobilenet_v2, densenet121, ResNet50_Weights, MobileNet_V2_Weights, DenseNet121_Weights


#model build
#optimizer
#metrics


class ModelWrapper:
    def __init__(self,
                 model_name: str,
                 opt_name: str,
                 num_classes: int,
                 learning_rate: float,
                 weight_decay: float,
                 multi_label: bool = True,
                 focal_loss: bool = True):

        self.model_name = model_name
        self.opt_name = opt_name
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.focal_loss = focal_loss

    def backbone(self):

        if self.model_name == "resnet50":

            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            resnet_in = model.fc.in_features
            model.fc = nn.Linear(in_features=resnet_in, out_features=self.num_classes)

            return model

        elif self.model_name == "mobilenetv2":

            model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            mobile_in = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features=mobile_in, out_features=self.num_classes)

            return model

        elif self.model_name == "densenet121":

            model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            dense_in = model.classifier.in_features
            model.classifier = nn.Linear(in_features=dense_in, out_features=self.num_classes)

            return model
        
        else:
            raise ValueError(f"Unkown model: {self.model_name} was given!")

    def model_loss(self):

        if self.multi_label:

            if self.focal_loss:

                loss = sigmoid_focal_loss()
            else:

                loss = nn.BCEWithLogitsLoss()
        else:

            loss = nn.CrossEntropyLoss()

        return loss

    def activation(self):

        if self.multi_label:

            return lambda x: torch.sigmoid(x)
        else:

            return lambda x: torch.softmax(x, dim=1)

    def build_optimizer(self, model):
        
        if self.opt_name.lower() == "adam":
            opt = optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.opt_name.lower() == "adamw":
            opt = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        elif self.opt_name.lower() == "sgd":
            opt = optim.SGD(model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unkown optimizer: {self.opt_name} was given!")

        return opt

    def build_model(self):

        model = self.backbone()
        loss = self.model_loss()
        act = self.activation()
        opt = self.build_optimizer(model)

        return model, loss, act, opt