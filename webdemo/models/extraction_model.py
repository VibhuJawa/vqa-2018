import torch.nn as nn
from torchvision import models

def get_pretrained_model(architecture="resnet152", cuda=False, data_parallel=False):
    """

    Args:
        architecture: Selects pretrained architecture, wraps around in a model with user defined forward functioon
        cuda: Allows using GPU
        data_parallel: Allows data parallelism

    Returns: Wrapped model with user defined forward pass and pretrained network weights

    Raises:
        ValueError : If data_parallel is true and cuda is False
    """

    class WrapModel(nn.Module):
        """
        Allows you to wrap any pre defined model with a pre defined user function
        """

        def __init__(self, net, forward_fn):

            super(WrapModel, self).__init__()
            self.net = net
            self.forward_fn = forward_fn

        def forward(self, x):
            return self.forward_fn(self.net, x)

        def __getattr__(self, attr):
            try:
                return super(WrapModel, self).__getattr__(attr)
            except AttributeError:
                return getattr(self.net, attr)

    def forward_resnet(self, x):
        """
        Args:
            self: The original model
            x: The input passed to the forward function

        Returns:
            x : Of dimension batch size * 2048 * 14 * 14
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_resnet2(self, x):
        """

        Args:
            self: The original model
            x: The input passed to the forward function

        Returns:
            x : Of dimension batch size * 2048
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        div = x.size(3) + x.size(2)
        x = x.sum(3)
        x = x.sum(2)
        x = x.view(x.size(0), -1)
        x = x.div(div)
        return x

    if architecture == "resnet152":
        predefined = models.resnet152(pretrained=True)
        model = WrapModel(predefined, forward_resnet)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model
