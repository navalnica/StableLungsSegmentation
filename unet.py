from model_blocks import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def print_tensor_size(self, tensor):
        # print('tensor size: ', tensor.size())
        pass

    def forward(self, x):
        self.print_tensor_size(x)

        x1 = self.inc(x)
        self.print_tensor_size(x1)

        x2 = self.down1(x1)
        self.print_tensor_size(x2)

        x3 = self.down2(x2)
        self.print_tensor_size(x3)

        x4 = self.down3(x3)
        self.print_tensor_size(x4)

        x5 = self.down4(x4)
        self.print_tensor_size(x5)

        x = self.up1(x5, x4)
        self.print_tensor_size(x)

        x = self.up2(x, x3)
        self.print_tensor_size(x)

        x = self.up3(x, x2)
        self.print_tensor_size(x)

        x = self.up4(x, x1)
        self.print_tensor_size(x)

        x = self.outc(x)
        self.print_tensor_size(x)

        return torch.sigmoid(x)
