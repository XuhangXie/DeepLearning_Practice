from torch import nn
import math
import torch.utils.model_zoo as model_zoo


# down sample
expansion = 4
def down_sample(inplanes,stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, inplanes*expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(inplanes*expansion)
    )

class bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1,downsample=None):
        super(bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(x)

        out = self.conv2(x)
        out = self.bn2(x)

        out = self.conv3(x)
        out = self.bn3(x)

        if self.downsample is not None:
            out = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Res50(nn.Module):

    def __init__(self, num_classes):
        super(Res50, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layer1 stride = 1, for max pooling already done it
        self.layer1 = self._make_layer(bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(bottleneck, 512, 3, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = down_sample(planes,stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# Test
model = Res50(num_classes=4)
print(model)

# if wants pre_trained weights
pretrained = False
if pretrained:
    model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))