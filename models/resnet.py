import torch.nn as nn


__all__ = ['ResNet101', 'ResNet152']


class ResNet(nn.Module):

    def __init__(self, resnet_version='resnet50', num_classes=10):
        super(ResNet, self).__init__()
        from torchvision.models.resnet import resnet18, \
            resnet34, resnet50, resnet101, resnet152
        if resnet_version == 'resnet18':
            resnet = resnet18(pretrained=True)
        elif resnet_version == 'resnet34':
            resnet = resnet34(pretrained=True)
        elif resnet_version == 'resnet50':
            resnet = resnet50(pretrained=True)
        elif resnet_version == 'resnet101':
            resnet = resnet101(pretrained=True)
        elif resnet_version == 'resnet152':
            resnet = resnet152(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AvgPool2d(7)
        self.classifier = nn.Linear(512 * 4, 1000)

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
        x = self.classifier(x)
        return x

    def set_fine_tune_level(self, level=1, random_seed=0):

        if level == 1:
            layers = [self.conv1, self.bn1,
                      self.layer1, self.layer2,
                      self.layer3, self.layer4]
        elif level == 2:
            layers = [self.conv1, self.bn1,
                      self.layer1, self.layer2,
                      self.layer3]
        elif level == 3:
            layers = [self.conv1, self.bn1,
                      self.layer1, self.layer2]
        else:
            raise NotImplementedError('level should be eigther 1, 2 or 3')
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

        if random_seed > 0:
            import torch
            import math
            torch.manual_seed(random_seed)
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print('randomize layer: %s' % name)
                    if name.find('conv') >= 0:
                        n = param.data.size(2) \
                            * param.data.size(3) \
                            * param.data.size(0)
                        param.data.normal_(0, math.sqrt(2. / n))
                    elif name.find('bn') >= 0:
                        if name.find('weight') >= 0:
                            param.data.fill_(1)
                        else:
                            param.data.zero_()


def ResNet152(nClasses=1000):
    return ResNet(resnet_version='resnet152',
                  nClasses=nClasses)


def ResNet101(nClasses=1000):
    return ResNet(resnet_version='resnet101',
                  nClasses=nClasses)
