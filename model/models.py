import torch.nn as nn


class Model(nn.Module):
    """Some Information about Model"""

    def __init__(self):
        super(Model, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels=200000, out_channels=1000,
                      kernel_size=8, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            
            nn.Conv1d(in_channels=1000, out_channels=200000,
                      kernel_size=8, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(4, 25000, 4, stride=2, padding=1),
            # nn.InstanceNorm2d(128),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(20, 1000, 4, stride=2, padding=1),
            # nn.InstanceNorm2d(256),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(256, 512, 4, padding=1),
            # nn.InstanceNorm2d(512),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    """Some Information about Discriminator"""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):

        return self.main(x)


class Generator(nn.Module):
    """Some Information about Generator"""

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Downsampling
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # # Residual blocks
            # ResidualBlock(256),
            # ResidualBlock(256),
            # ResidualBlock(256),
            # ResidualBlock(256),
            # ResidualBlock(256),
            # ResidualBlock(256),
            # ResidualBlock(256),
            # ResidualBlock(256),
            # ResidualBlock(256),

            # Upsampling
            nn.ConvTranspose2d(256, 128, 3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )

    def forward(self, x):

        return self.main(x)
