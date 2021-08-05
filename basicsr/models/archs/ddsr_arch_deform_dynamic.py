import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.models.archs.arch_util import (DCNv2Pack, ResidualBlockNoBN, Upsample,
                                            make_layer)
import math


# 把动态卷积加到可形变卷积之中
class PixelConv(nn.Module):
    def __init__(self, scale=1, depthwise=False, stride=1):
        super().__init__()
        self.scale = scale
        self.depthwise = depthwise
        self.stride = stride

    def forward(self, feature, kernel):
        NF, CF, HF, WF = feature.size()
        NK, ksize, HK, WK = kernel.size()
        # assert NF == NK and HF == HK and WF == WK
        if self.depthwise:
            ink = CF
            outk = 1
            ksize = int(math.sqrt(int(ksize // (self.scale ** 2))))
            pad = (ksize - 1) // 2
        else:
            ink = 1
            outk = CF
            ksize = int(math.sqrt(int(ksize // CF // (self.scale ** 2))))
            pad = (ksize - 1) // 2
        # features unfold and reshape, same as PixelConv
        feat = F.pad(feature, [pad, pad, pad, pad])
        feat = feat.unfold(2, ksize, self.stride).unfold(3, ksize, self.stride)
        feat = feat.permute(0, 2, 3, 1, 5, 4).contiguous()
        feat = feat.reshape(NF, HF // self.stride, WF // self.stride, ink, -1)

        # kernel
        kernel = kernel.permute(0, 2, 3, 1).reshape(NK, HK, WK, ksize * ksize, self.scale ** 2 * outk)
        # 16, 80, 45, 3, 25] [16, 240, 135, 25, 1]
        output = torch.matmul(feat, kernel)
        output = output.permute(0, 3, 4, 1, 2).view(NK, -1, HF // self.stride, WF // self.stride)
        if self.scale > 1:
            output = F.pixel_shuffle(output, self.scale)
        return output


# 每一个block的输入是图像和backbone提供的特征，使用动态卷积得到offset
class DynamicConvBlock(nn.Module):
    def __init__(self, inc=3, num_feat=64, kernel_size=5):
        super().__init__()
        self.input_conv = nn.Sequential(nn.Conv2d(inc, num_feat, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.feat_residual = nn.Sequential(nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1), nn.ReLU(inplace=True),
                                           nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.feat_kernel = nn.Conv2d(2 * num_feat, kernel_size * kernel_size, 3, stride=1, padding=1)
        self.pixel_conv = PixelConv(scale=1, depthwise=True, stride=1)

    # 根据
    def forward(self, input_tensor, feature):  # image 是当前尺度的输入(32) features是深层特征(128),对输入的image进行卷积
        input_tensor = self.input_conv(input_tensor)
        cat_inputs = torch.cat([input_tensor, feature], 1)
        kernel = self.feat_kernel(cat_inputs)
        output = self.pixel_conv(input_tensor, kernel)
        residual = self.feat_residual(cat_inputs)
        return output + residual


class feature_extract(nn.Module):
    def __init__(self, num_feat):
        super(feature_extract, self).__init__()
        self.first_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.res_block = make_layer(ResidualBlockNoBN, 5, num_feat=num_feat)

    def forward(self, x):  # 输入是图像
        x = self.first_conv(x)
        res = self.res_block(x)
        return x + res


class refine_blocks(nn.Module):
    def __init__(self):
        super(refine_blocks, self).__init__()
        self.dynamic_block_1 = DynamicConvBlock(inc=64, num_feat=64, kernel_size=5)
        # self.dynamic_block_2 = DynamicConvBlock(inc=64, num_feat=64, kernel_size=5)
        # self.dynamic_block_3 = DynamicConvBlock(inc=64, num_feat=64, kernel_size=5)
        # self.dynamic_block_4 = DynamicConvBlock(inc=64, num_feat=64, kernel_size=5)
        # self.dynamic_block_5 = DynamicConvBlock(inc=64, num_feat=64, kernel_size=5)

    def forward(self, img_tensor, feature):
        # print('feature_1', img_tensor.shape, feature.shape)
        feature_1 = self.dynamic_block_1(img_tensor, feature)
        # feature_2 = self.dynamic_block_2(feature_1, feature)
        # feature_3 = self.dynamic_block_3(feature_2, feature)
        # feature_4 = self.dynamic_block_4(feature_3, feature)
        # feature_5 = self.dynamic_block_5(feature_4, feature)
        # return feature_5
        return feature_1


class Dynamic_convNet(nn.Module):
    def __init__(self, num_feat=64):
        super(Dynamic_convNet, self).__init__()
        self.FEX = feature_extract(num_feat=num_feat)
        self.REF = refine_blocks()

    def forward(self, x):
        # print('input', x.shape)
        feature = self.FEX(x)
        return self.REF(x, feature)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB,
            num_block,  # 20
            num_feat=num_feat,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x


# 分成两块，特征提取和重建部分
# 特征提取分成两个支线，多尺度上的形变卷积和动态卷积
class Muti_scale_extractor(nn.Module):
    def __init__(self, in_channel=3, num_feat=64, deformable_groups=8):
        super(Muti_scale_extractor, self).__init__()
        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()  # 键值对
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()  # 每层都有一个DCN，都是一样的，根据offset进行卷积
        self.feat_conv = nn.ModuleDict()
        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'  # l3 l2 l1
            # 第一个卷积层使用动态卷积
            self.offset_conv1[level] = Dynamic_convNet(num_feat=64)
            # self.offset_conv1[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 两帧concat作为偏移的输入
            # TODO: concat核的信息
            if i == 3:  # 最后一层offset只是由两帧cat得到的
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                # 其他层上还有上一层的偏移的上采样
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3,
                                                     1, 1)  # concat
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)  # 通道转换
            self.dcn_pack[level] = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
            if i < 3:  # 下一层特征与当前层DCN输出经过拼接融合
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=num_feat)
        self.conv_first_1 = nn.Conv2d(in_channel, num_feat, 3, 1, 1)
        # self.conv_first_2 = nn.Conv2d(in_channel, num_feat, 3, 1, 1)
        # self.conv_first_3 = nn.Conv2d(in_channel, num_feat, 3, 1, 1)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)  # 使用2步长卷积下采样
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 每层还有一个正常的卷积
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, image_tensor):
        feat_l1 = self.lrelu(self.conv_first_1(image_tensor))
        # feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        # feat_l2 = F.interpolate(image_tensor, size=None, scale_factor=0.5, mode='bicubic', align_corners=False)
        # feat_l2 = self.lrelu(self.conv_first_2(feat_l2))
        #
        # feat_l3 = F.interpolate(image_tensor, size=None, scale_factor=0.25, mode='bicubic', align_corners=False)
        # feat_l3 = self.lrelu(self.conv_first_3(feat_l3))
        # print('layer shape', image_tensor.shape, feat_l2.shape, feat_l3.shape)
        feat_l = [feat_l1, feat_l2, feat_l3]
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            # ######################################offset############################################
            level = f'l{i}'
            offset = feat_l[i - 1]  # 每一层的输入特征改成直接下采样
            # 每一层使用动态卷积得到offset,使用特征提取网络和多层的动态卷积层
            offset = self.lrelu(self.offset_conv1[level](offset))  # offset_conv1
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))  # 加上层的上采样
                offset = self.lrelu(self.offset_conv3[level](offset))  # 转换通道
            # print('fucking', feat_l[i - 1].shape, offset.shape)
            # ######################################feature############################################
            feat = self.dcn_pack[level](feat_l[i - 1], offset)  # DCN的输入就是向量和偏移量，这个偏移量不是二通道，
            # 而是输入的特征，正真的offset封装在里面
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)
            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)
        # Cascading
        offset = torch.cat([feat, feat_l[0]], dim=1)
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class PredeblurModule(nn.Module):
    """Pre-dublur module.

    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        hr_in (bool): Whether the input has high resolution. Default: False.
    """

    def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
        super(PredeblurModule, self).__init__()
        self.hr_in = hr_in

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        if self.hr_in:
            # downsample x4 by stride conv
            self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
            self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        # generate feature pyramid 特征金字塔
        self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l1 = nn.ModuleList(
            [ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        feat_l1 = self.lrelu(self.conv_first(x))
        if self.hr_in:
            feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))
            feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))

        # generate feature pyramid
        feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))
        feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))

        feat_l3 = self.upsample(self.resblock_l3(feat_l3))
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
        feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))

        for i in range(2):
            feat_l1 = self.resblock_l1[i](feat_l1)
        feat_l1 = feat_l1 + feat_l2
        for i in range(2, 5):
            feat_l1 = self.resblock_l1[i](feat_l1)
        return feat_l1


class DDSR(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 with_predeblur=False,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 img_range=255.):
        super(DDSR, self).__init__()
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.img_range = img_range
        self.with_predeblur = with_predeblur
        # extract features for each frame
        if self.with_predeblur:
            self.predeblur = PredeblurModule(num_feat=num_feat, hr_in=self.hr_in)
        # reconstruction
        self.feature_extractor = Muti_scale_extractor(in_channel=num_in_ch)
        self.reconstruction = make_layer(ResidualGroup, 10, num_feat=64, num_block=20, squeeze_factor=16, res_scale=1)
        # self.reconstruction = make_layer(ResidualBlockNoBN, 10, num_feat=num_feat)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(4, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.feature_extractor(x)
        feat = self.reconstruction(x)
        res = self.conv_after_body(feat)
        res += x
        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean
        return x

# if __name__ == '__main__':
#     net = DDSR().cuda().half()
#     img_tensor = torch.randn(16, 3, 320, 180).cuda()
#     print(net(img_tensor.type(torch.float16)).shape)
