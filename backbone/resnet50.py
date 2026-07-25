from collections import OrderedDict

from torch import nn
from torchvision.models import resnet50


class ResNet50(nn.Module):
    def __init__(self, freeze_at=0, pretrained=False):
        super().__init__()
        weights = None
        self.body = resnet50(weights=weights)
        self.stem = nn.Sequential(
            self.body.conv1,
            self.body.bn1,
            self.body.relu,
            self.body.maxpool,
        )
        self.layer1 = self.body.layer1
        self.layer2 = self.body.layer2
        self.layer3 = self.body.layer3
        self.layer4 = self.body.layer4
        self.out_channels = [256, 512, 1024, 2048]
        self.freeze(freeze_at)

    def freeze(self, freeze_at):
        blocks = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for block in blocks[: max(0, int(freeze_at))]:
            block.eval()
            for param in block.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return OrderedDict(c2=c2, c3=c3, c4=c4, c5=c5)
