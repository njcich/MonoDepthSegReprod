
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models


class DepthEncoder(nn.Module):
    def __init__(self):
        super(DepthEncoder, self).__init__()
        self.encoder = models.resnet50(pretrained=True)
        self.num_ch_enc = np.array([64, 256, 512, 1024, 2048])

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
