from pickletools import uint8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import SWT


class LSTM(nn.Module):
    def __init__(self,n_feats):
        super(LSTM,self).__init__()

        self.iteration = 3

        self.conv_f = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max1 = nn.AdaptiveMaxPool2d((1, 1))
        self.conv3 = nn.Conv2d(n_feats * 2, n_feats * 2, 3, 1, 1)
        self.max2 = nn.AdaptiveMaxPool2d((1, 1))
        self.conv4 = nn.Conv2d(n_feats * 2, n_feats * 2, 3, 1, 1)

        self.convxh = torch.nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, stride=1, padding='same')

        self.convfc = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.convfcmax = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.convfcout = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        self.convig = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.convigmax = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.convigout = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        self.attention_out = Attention(n_feats)


    def forward(self,x,h,c):
        xin = x

        for i in range(self.iteration):
            feature = torch.cat((xin, h), 1)

            f = self.conv_f(feature)
            g = self.conv_g(feature)
            i = self.conv_i(feature)
            o = self.conv_o(feature)
            # c = f * c + i * g

            fc = f * c
            fc1 = self.convfc(fc)
            fc_max = self.max1(fc1)
            fc_max = self.convfcmax(fc_max)
            fc = fc1 + fc_max
            fc = self.convfcout(fc)

            ig = i * g
            ig1 = self.convig(ig)
            ig_max = self.max2(ig1)
            ig_max = self.convigmax(ig_max)
            ig = ig1 + ig_max
            ig = self.convigout(ig)

            c = fc + ig
            h = o * torch.tanh(c)  # h 일단 그대로

            xin = self.attention_out(h)

        return xin, h, c



class detail(nn.Module):  # 변경함
    def __init__(self, n_feats):
        super(detail, self).__init__()

        self.swt = SWT.SWTForward(J=1, wave='haar', mode='symmetric')

        self.max1 = nn.AdaptiveMaxPool2d((1, 1))
        self.max2 = nn.AdaptiveMaxPool2d((1, 1))
        self.max3 = nn.AdaptiveMaxPool2d((1, 1))

        self.convh = torch.nn.Conv2d(n_feats * 3, n_feats, kernel_size=3, stride=1, padding='same')
        self.convfinal = torch.nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding='same')

    def waveH(self, t1, t2, t3):
        #Separate

        t1_0, t1_1, t1_2, t1_3 = torch.chunk(t1[0], 4, dim=1)
        t2_0, t2_1, t2_2, t2_3 = torch.chunk(t2[0], 4, dim=1)
        t3_0, t3_1, t3_2, t3_3 = torch.chunk(t3[0], 4, dim=1)
        lh = torch.cat((t1_1, t2_1, t3_1), dim=1)
        hl = torch.cat((t1_2, t2_2, t3_2), dim=1)
        hh = torch.cat((t1_3, t2_3, t3_3), dim=1)
        total = torch.cat((lh, hl, hh), dim=1)

        return total

    def forward(self, x):

        r, g, b = torch.split(x, 1, dim=1)
        red = self.swt(r)
        green = self.swt(g)
        blue = self.swt(b)

        x_high = self.waveH(red, green, blue)

        lh = x_high[:, 0:3]
        hl = x_high[:, 3:6]
        hh = x_high[:, 6:9]

        mh = self.max1(lh)
        mv = self.max2(hl)
        md = self.max3(hh)
        lh = lh * mh
        hl = hl * mv
        hh = hh * md
        high = torch.cat((lh, hl, hh), 1)
        high = self.convh(high)

        out = x + high
        out = self.convfinal(out)

        return out

class ConvPReLU(nn.Module):
    def __init__(self,in_ch, out_ch, kernel_size, stride=1, padding='same'):
        super(ConvPReLU,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding= padding),
            nn.PReLU(),
        )

    def forward(self,x):
        xout = self.conv(x)

        return xout

class Attention(nn.Module):
    def __init__(self,in_channel):
        super(Attention,self).__init__()

        self.conv3 = torch.nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=1,padding='same')
        self.conv5 = torch.nn.Conv2d(in_channel, in_channel, kernel_size=5, stride=1, padding='same')
        self.conv7 = torch.nn.Conv2d(in_channel, in_channel, kernel_size=7, stride=1, padding='same')

        self.max3 = nn.AdaptiveMaxPool2d((1, 1))
        self.max5 = nn.AdaptiveMaxPool2d((1, 1))
        self.max7 = nn.AdaptiveMaxPool2d((1, 1))

        self.conv3m = torch.nn.Conv2d(in_channel,in_channel,kernel_size=1,stride=1,padding='same')
        self.conv5m = torch.nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding='same')
        self.conv7m = torch.nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding='same')

        self.convf = ConvPReLU(in_channel*3 , in_channel,3,1,'same')

    def forward(self,x):

        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        x3w = self.max3(x3)
        x5w = self.max5(x5)
        x7w = self.max7(x7)

        x3w = self.conv3m(x3w)
        x5w = self.conv5m(x5w)
        x7w = self.conv7m(x7w)

        x3 = x3 + x3w
        x5 = x5 + x5w
        x7 = x7 + x7w

        xout = torch.cat((x3,x5,x7),1)

        xout = self.convf(xout)

        return xout


class SF(nn.Module):
    def __init__(self):
        super(SF,self).__init__()
        # self.conv3to64 = torch.nn.Conv2d(x_feat, 64, kernel_size=3, stride=1, padding='same')
        self.slConv6to3 = torch.nn.Conv2d(6, 3, kernel_size=3, stride=1, padding='same')
        self.max = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self,x,sat):
        # x = self.conv3to64(x)  # 64
        sat_low = torch.cat((x, sat), dim=1)  # 6
        sat_low_max = self.max(sat_low)
        sat_low_weighted6 = sat_low * sat_low_max
        sat_low_weighted3 = self.slConv6to3(sat_low_weighted6)
        x_sat = x * sat_low_weighted3  #

        return x_sat


class LIENet(nn.Module):
    def __init__(self,in_dim=3,use_GPU=True):
        super(LIENet,self).__init__()

        self.attention = Attention(3)
        self.convSatToH = torch.nn.Conv2d(1, 64, kernel_size=1, stride=1, padding='same')

        self.conv3to32 = torch.nn.Conv2d(in_dim, 32, kernel_size=3, stride=1, padding='same')
        self.conv32to64 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same')
        self.conv64to32 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='same')
        self.conv32to3 = ConvPReLU(32,3,3,1,'same')

        self.lstm1 = LSTM(n_feats=64)

        # self.sharpening = detail(3)  # detail 제거

        self.conv_final = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding='same')

        self.use_GPU = use_GPU



    def rgb2hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)

        return hsv_s  # 1 (값 범위 :0~1)

    def forward(self,x_ori):

        xin = x_ori
        xin = self.attention(xin)

        sat = self.rgb2hsv(x_ori)

        x = self.conv3to32(x_ori)
        x = self.conv32to64(x)

        batch, channel, width, height = x_ori.shape
        c0 = torch.zeros((batch, 64, width, height))
        h0 = self.convSatToH(sat)

        if self.use_GPU:
            c0 = c0.cuda()

        x1, h1, c1 = self.lstm1(x, h0, c0)


        out = self.conv64to32(x1)
        out = self.conv32to3(out) #3

        out = xin + out

        # out = self.sharpening(out) # detail block 제거

        pred = self.conv_final(out)


        return pred









###################################################################



# ULBP
class ULBP(nn.Module):
    def __init__(self, inchannel):
        super(ULBP, self).__init__()

        # Encoder
        self.enc1_1 = ConvBlock(input_size=inchannel, output_size=64, kernel_size=3, stride=1, padding='same')
        self.enc1_2 = ConvBlock(input_size=64, output_size=64, kernel_size=3, stride=1, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = ConvBlock(input_size=64, output_size=128, kernel_size=3, stride=1, padding='same')
        self.enc2_2 = ConvBlock(input_size=128, output_size=128, kernel_size=3, stride=1, padding='same')

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = ConvBlock(input_size=128, output_size=256, kernel_size=3, stride=1, padding='same')
        self.enc3_2 = ConvBlock(input_size=256, output_size=256, kernel_size=3, stride=1, padding='same')

        # Decoder
        self.unpool3 = upsample(input_size=256, output_size=128, kernel_size=2, stride=2, padding=0)
        self.dec4_1 = ConvBlock(input_size=256, output_size=256, kernel_size=3, stride=1, padding='same')
        self.dec4_2 = ConvBlock(input_size=256, output_size=128, kernel_size=3, stride=1, padding='same')

        self.unpool4 = upsample(input_size=128, output_size=64, kernel_size=2, stride=2, padding=0)
        self.dec5_1 = ConvBlock(input_size=128, output_size=64, kernel_size=3, stride=1, padding='same')
        self.dec5_2 = ConvBlock(input_size=64, output_size=64, kernel_size=3, stride=1, padding='same')

        # LBP, MS, No
        self.lbp = LBP(input_size=256, output_size=256, kernel_size=3, stride=1, padding='same')
        self.ms = MSBlock(input_size=128, output_size=128, kernel_size=3)
        self.no = NoiseBlock(input_size=64, output_size=64, kernel_size=3)
        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)  # 수정

    def forward(self, x):
        # ULBP

        feature1 = self.enc1_1(x)  # 1 64
        feature1 = self.enc1_2(feature1)
        pool_feature1 = self.pool1(feature1)

        feature2 = self.enc2_1(pool_feature1)  # 1/2 128
        feature2 = self.enc2_2(feature2)
        pool_feature2 = self.pool1(feature2)

        feature3 = self.enc3_1(pool_feature2)  # 1/4 256

        LBP_feature3 = self.lbp(feature3)  # 1/4 256

        feature4 = self.unpool3(LBP_feature3)  # 1/2 128

        feature4 = torch.cat((feature2, feature4), 1)  # 1/2 256
        feature4 = self.dec4_1(feature4)  # 1/2 256 -> 128
        feature4 = self.dec4_2(feature4)  # 1/2 128 -> 128

        MS_feature4 = self.ms(feature4)  # 1/2 128

        feature5 = self.unpool4(MS_feature4)  # 1 64
        feature5 = torch.cat((feature1, feature5), 1)  # 1 128
        feature5 = self.dec5_1(feature5)  # 1/2 128 -> 64
        feature5 = self.dec5_2(feature5)  # 1/2 64 -> 64

        NO_feature6 = self.no(feature5)
        ULBP_out = self.out(NO_feature6)

        return ULBP_out


# LBP


class LightenBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(LightenBlock, self).__init__()
        self.conv_Encoder = ASB(input_size, output_size, kernel_size=3)
        self.conv_Offset = ASB(input_size, output_size, kernel_size=3)
        self.conv_Decoder = ASB(input_size, output_size, kernel_size=3)

    def forward(self, x):
        offset = self.conv_Offset(x)
        code_lighten = x + offset
        out = self.conv_Decoder(code_lighten)
        return out


class DarkenBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DarkenBlock, self).__init__()
        self.conv_Encoder = ASB(input_size, output_size, kernel_size=3)
        self.conv_Offset = ASB(input_size, output_size, kernel_size=3)
        self.conv_Decoder = ASB(input_size, output_size, kernel_size=3)

    def forward(self, x):
        offset = self.conv_Offset(x)
        code_lighten = x - offset
        out = self.conv_Decoder(code_lighten)
        return out


class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(FusionLayer, self).__init__()

        self.MASE = MASEblock(inchannel)
        self.PASB = ASB(inchannel, outchannel, kernel_size=3)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.MASE(x).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y + x
        y = self.PASB(y)
        return y


class LBP(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(LBP, self).__init__()
        self.fusion = FusionLayer(input_size, output_size)
        self.conv1 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DarkenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1_1 = weASB(input_size, output_size, kernel_size=1)
        self.local_weight2_1 = weASB(input_size, output_size, kernel_size=1)

    def forward(self, x):
        x = self.fusion(x)
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1_1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2_1(hr)
        return hr_weight + h_residue


# Multi Scale / Noise Block


class MSBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size):
        super(MSBlock, self).__init__()
        self.ds1 = nn.Conv2d(input_size, output_size, kernel_size, stride=2, padding=1)
        self.ds2 = nn.Conv2d(input_size, output_size, kernel_size, stride=2, padding=1)
        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv2 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv3 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv4 = nn.Conv2d(input_size * 4, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.bn1 = nn.BatchNorm2d(input_size)
        self.bn2 = nn.BatchNorm2d(input_size)
        self.bn3 = nn.BatchNorm2d(input_size)
        self.act1 = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.act1(out1)  # CONV BLOCK

        out2 = self.ds1(x)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        out2 = self.act1(out2)
        out2 = F.interpolate(out2, size=(out1.size()[2], out1.size()[3]))

        out3 = self.ds1(x)
        out3 = self.ds2(out3)
        out3 = self.conv3(out3)
        out3 = self.bn3(out3)
        out3 = self.act1(out3)
        out3 = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        out3 = F.interpolate(out3, size=(out1.size()[2], out1.size()[3]))
        out = torch.cat([x, out1, out2, out3], dim=1)
        out = self.conv4(out)

        return out


class NoiseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size):
        super(NoiseBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv2 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv3 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.act1 = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.act1(out1)
        out2 = self.conv2(out1)
        out2 = self.act1(out2)
        out3 = self.conv3(out2)

        out = torch.tanh(out3)

        return out


# Base modules


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out


class upsample(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding='same'):
        super(upsample, self).__init__()
        self.up = nn.ConvTranspose2d(input_size, output_size, kernel_size=2, stride=2, padding=0, bias=True)

    def forward(self, x):  # skip 빈거 넣어줄때
        x = self.up(x)

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# Stretching Block(ASB/ANB)


class MASEblock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveMaxPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x


class MISEblock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveMaxPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = -self.squeeze(-x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x


class ANB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.maseblock = MASEblock(in_channels)
        self.miseblock = MISEblock(in_channels)

    def forward(self, x):
        im_h = self.maseblock(x)
        im_l = self.miseblock(x)

        me = torch.tensor(0.00001, dtype=torch.float32).cuda()

        x = (x - im_l) / torch.maximum(im_h - im_l, me)
        x = torch.clip(x, 0.0, 1.0)

        return x


class ASB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class weASB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.conv(x)

        return x
