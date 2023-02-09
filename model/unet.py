# code hongtao
import torch
import torch.nn as nn



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 1, padding_mode="zeros"),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_ch, channels=16, n_class=1):
        super(self, Unet).init()
        self.conv1 = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.conv2 = DoubleConv(channels, channels * 2)
        self.conv3 = DoubleConv(channels * 2, channels * 4)
        self.conv4 = DoubleConv(channels * 4, channels * 8)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.out = nn.Conv2d(channels, n_class, 1)

    def forward(self, input):
        c1 = self.conv1(input)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

        up_5 = self.up5(c4)
        merge5 = torch.cat([up_5, c3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1], dim=1)
        c7 = self.conv7(merge7)
        out = self.out(c7)
        return out


if __name__ == "__main__":
    print(Unet(1,1))