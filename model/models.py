import torch.nn as nn
import torch
import torchvision


class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size,
                               stride=stride, dilation=dilation, padding=3, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)

        return out


class se_block(nn.Module):
    def __init__(self, in_layer, out_layer):
        super(se_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer//8,
                               kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer//8, in_layer,
                               kernel_size=1, padding=0)
        self.fc = nn.Linear(1, out_layer//8)
        self.fc2 = nn.Linear(out_layer//8, out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_se = nn.functional.adaptive_avg_pool1d(x, 1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)

        x_out = torch.add(x, x_se)
        return x_out


class re_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()

        self.cbr1 = conbr_block(in_layer, out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer, out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)

    def forward(self, x):

        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out


class UNET_1D(nn.Module):
    def __init__(self, input_dim, layer_n, kernel_size, depth):
        super(UNET_1D, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth

        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=5)
        self.AvgPool1D2 = nn.AvgPool1d(input_dim, stride=25)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=125)

        self.layer1 = self.down_layer(
            self.input_dim, self.layer_n, self.kernel_size, 1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(
            self.layer_n*2), self.kernel_size, 5, 2)
        self.layer3 = self.down_layer(int(
            self.layer_n*2)+int(self.input_dim), int(self.layer_n*3), self.kernel_size, 5, 2)
        self.layer4 = self.down_layer(int(
            self.layer_n*3)+int(self.input_dim), int(self.layer_n*4), self.kernel_size, 5, 2)
        self.layer5 = self.down_layer(int(
            self.layer_n*4)+int(self.input_dim), int(self.layer_n*5), self.kernel_size, 4, 2)

        self.cbr_up1 = conbr_block(
            int(self.layer_n*7), int(self.layer_n*3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(
            int(self.layer_n*5), int(self.layer_n*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(
            int(self.layer_n*3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=5, mode='nearest')

        self.outcov = nn.Conv1d(
            self.layer_n, 11, kernel_size=self.kernel_size, stride=1, padding=3)

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer, out_layer, kernel, 1))
        return nn.Sequential(*block)

    def forward(self, x):

        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)

        #############Encoder#####################

        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)

        x = torch.cat([out_1, pool_x1], 1)
        out_2 = self.layer3(x)

        x = torch.cat([out_2, pool_x2], 1)
        x = self.layer4(x)

        #############Decoder####################

        up = self.upsample1(x)
        up = torch.cat([up, out_2], 1)
        up = self.cbr_up1(up)

        up = self.upsample(up)
        up = torch.cat([up, out_1], 1)
        up = self.cbr_up2(up)

        up = self.upsample(up)
        up = torch.cat([up, out_0], 1)
        up = self.cbr_up3(up)

        out = self.outcov(up)

        #out = nn.functional.softmax(out,dim=2)

        return out


class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, 3),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.main(x)


class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecBlock, self).__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 3)
        self.down = nn.Sequential(
            nn.Conv1d(out_ch * 2, out_ch, 3),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def crop(self, x, enc_ftrs):
        C, H, W = x.shape
        # print(C, H, W)
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

    def forward(self, x, encoder_features):
        x = self.up(x)
        enc_ftrs = self.crop(x, encoder_features)
        x = torch.cat([x, enc_ftrs], dim=1)
        # print('x:', x.shape, 'f:', enc_ftrs.shape,
        #       'fo:', encoder_features.shape)
        x = self.down(x)
        return x


# class Encoder(nn.Module):
#     def __init__(self, chs=(1, 3, 64, 128, 256, 512)):
#         super().__init__()
#         self.enc_blocks = nn.ModuleList(
#             [EncBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
#         # self.pool = nn.MaxPool1d(2)

#         # self.main = nn.Sequential(
#         #     nn.Conv1d(1, 3, 12),
#         #     nn.BatchNorm1d(3),
#         #     nn.LeakyReLU(0.2),
#         #     nn.Conv1d(3, 64, 12),
#         #     nn.BatchNorm1d(64),
#         #     nn.LeakyReLU(0.2),
#         #     nn.Conv1d(64, 128, 12),
#         #     nn.BatchNorm1d(128),
#         #     nn.LeakyReLU(0.2),
#         #     nn.Conv1d(128, 256, 12),
#         #     nn.BatchNorm1d(256),
#         #     nn.LeakyReLU(0.2)
#         # )

#     def forward(self, x):
#         return self.main(x)


# class Decoder(nn.Module):
#     def __init__(self, chs=(256, 128, 64, 3, 1)):
#         super().__init__()
#         self.chs = chs
#         # self.main= nn.Sequential(
#         #     nn.ConvTranspose1d(256, 128, 12),
#         #     nn.BatchNorm1d(128),
#         #     nn.LeakyReLU(0.2),
#         #     nn.ConvTranspose1d(128, 64, 12),
#         #     nn.BatchNorm1d(64),
#         #     nn.LeakyReLU(0.2),
#         #     nn.ConvTranspose1d(64, 3, 12),
#         #     nn.BatchNorm1d(3),
#         #     nn.LeakyReLU(0.2),
#         #     nn.ConvTranspose1d(3, 1, 12),
#         #     nn.BatchNorm1d(1),
#         #     nn.LeakyReLU(0.2)
#         # )
#         # self.upconvs = nn.ModuleList(
#         #     [nn.ConvTranspose1d(chs[i], chs[i+1], 2) for i in range(len(chs)-1)])
#         self.dec_blocks = nn.ModuleList(
#             [DecBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])

#     def forward(self, x, encoder_features):

#         for i in range(len(self.chs)-1):
#             # x = self.upconvs[i](x)
#             enc_ftrs = self.crop(encoder_features[i], x)
#             x = torch.cat([x, enc_ftrs], dim=1)
#             x = self.dec_blocks[i](x)
#         return x

#     def crop(self, enc_ftrs, x):
#         C, H, W = x.shape
#         print(C, H, W)
#         enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
#         return enc_ftrs


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = EncBlock(1, 3)
        self.enc1 = EncBlock(3, 64)
        self.enc2 = EncBlock(64, 128)
        self.enc3 = EncBlock(128, 256)
        self.enc4 = EncBlock(256, 512)
        self.dec1 = DecBlock(512, 256)
        self.dec2 = DecBlock(256, 128)
        self.dec3 = DecBlock(128, 64)
        self.dec4 = DecBlock(64, 3)
        self.out = nn.Conv1d(3, 1, 3)
        # for i in range(len(chs) - 1):
        #     self.main.append(EncBlock(chs[i], chs[i + 1]))

        # for i in reversed(range(len(chs) - 1)):
        #     self.main.append(DecBlock(chs[i + 1], chs[i]))

    def forward(self, x):
        inc = self.inc(x)
        x1 = self.enc1(inc)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x = self.dec1(x4, x3)
        x = self.dec2(x, x2)
        x = self.dec3(x, x1)
        x = self.dec4(x, inc)
        out = self.out(x)
        return out

        # enc = self.encoder(x)
        # print('enc:', enc.shape)
        # out = self.decoder(enc[::-1][0], enc[::-1][1:])
        # # out      = self.head(out)

        # # if self.retain_dim:
        # #     out = F.interpolate(out, out_sz)
        # return out


class Model(nn.Module):
    """Some Information about Model"""

    def __init__(self):
        super(Model, self).__init__()
        self.main = nn.Sequential(
            # nn.Conv1d(in_channels=200000, out_channels=1000,
            #           kernel_size=8, stride=2, padding=1),
            # nn.LeakyReLU(0.2, inplace=True),


            # nn.Conv1d(in_channels=1000, out_channels=200000,
            #           kernel_size=8, stride=2, padding=1),

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


            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, stride=1)
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
