import torch.nn as nn



class Discriminator(nn.Module):
    """Some Information about Discriminator"""

    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, x):

        return x


class Generator(nn.Module):
    """Some Information about Generator"""

    def __init__(self):
        super(Generator, self).__init__()

    def forward(self, x):

        return x
