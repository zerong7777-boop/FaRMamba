import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import fftpack
from .MSCA import AttentionModule  # MSCA 模块
from .CBAM import CBAM  # CBAM 模块


class DCTTransform(nn.Module):
    def __init__(self, img_size, num_freq_masks):
        """
        初始化 DCTTransform 模块
        :param img_size: 图像大小（假设为正方形）
        :param num_freq_masks: 使用的频域掩码数量
        """
        super(DCTTransform, self).__init__()
        self.img_size = img_size
        self.num_freq_masks = num_freq_masks
        self.freq_masks = self._generate_freq_masks(img_size, num_freq_masks)

    def _generate_freq_masks(self, img_size, num):
        """
        生成频域掩码
        :param img_size: 图像大小
        :param num: 掩码的数量
        :return: 生成的掩码列表
        """
        masks = []
        mask_none = torch.ones(img_size, img_size)  # 无掩码，保留全频域
        for i in reversed(range(num - 1)):
            mask_size = img_size // (2 ** i)
            mask = torch.ones(mask_size, mask_size)
            mask = torch.triu(mask, diagonal=0)  # 上三角部分
            mask = torch.rot90(mask, k=1, dims=(0, 1))  # 旋转掩码
            # 填充到 img_size × img_size
            mask = F.pad(mask, (0, img_size - mask_size, 0, img_size - mask_size), mode='constant', value=0)
            masks.append(mask)
        masks.append(mask_none)  # 添加无掩码
        return torch.stack(masks)  # 返回堆叠的掩码

    def forward(self, x):
        """
        前向传播方法
        :param x: 输入张量，形状为 (B, C, H, W)
        :return: LL, LH, HL, HH 频域张量
        """
        device = x.device
        batch_size, channels, height, width = x.shape

        # 确保掩码在设备上
        self.freq_masks = self.freq_masks.to(device)

        # 初始化频域张量
        LL_tensor = torch.zeros(batch_size, channels, height, width, device=device)
        LH_tensor = torch.zeros(batch_size, channels, height, width, device=device)
        HL_tensor = torch.zeros(batch_size, channels, height, width, device=device)
        HH_tensor = torch.zeros(batch_size, channels, height, width, device=device)

        # 对每个样本的每个通道进行 2D DCT
        for i in range(batch_size):
            for c in range(channels):
                dct_result = fftpack.dct(
                    fftpack.dct(x[i, c].detach().cpu().numpy(), axis=0, norm='ortho'),
                    axis=1,
                    norm='ortho',
                )

                # 将频域结果与掩码相乘，并应用 IDCT 还原
                for idx, mask in enumerate(self.freq_masks):
                    masked_dct = dct_result * mask.cpu().numpy()
                    idct_result = fftpack.idct(
                        fftpack.idct(masked_dct, axis=0, norm='ortho'),
                        axis=1,
                        norm='ortho',
                    )

                    # 存储还原后的张量
                    if idx == 0:
                        LH_tensor[i, c] = torch.tensor(idct_result, device=device)
                    elif idx == 1:
                        HL_tensor[i, c] = torch.tensor(idct_result, device=device)
                    elif idx == 2:
                        HH_tensor[i, c] = torch.tensor(idct_result, device=device)
                    else:
                        LL_tensor[i, c] = torch.tensor(idct_result, device=device)

        return LL_tensor, LH_tensor, HL_tensor, HH_tensor


class CombinedModule(nn.Module):
    def __init__(self, img_size=256, in_planes=48, num_freq_masks=4):
        super(CombinedModule, self).__init__()
        self.img_size = img_size
        self.initial_in_planes = in_planes

        # DCT 变换模块
        self.DCT = DCTTransform(img_size=img_size, num_freq_masks=num_freq_masks)

        # CBAM 模块
        self.CBAM_LH = CBAM(in_planes=in_planes)
        self.CBAM_HL = CBAM(in_planes=in_planes)
        self.CBAM_HH = CBAM(in_planes=in_planes)
        self.CBAM_LL = CBAM(in_planes=in_planes)

        # MSCA 模块
        self.MSCA_high_freq = AttentionModule(dim=in_planes * 3)  # 高频融合维度
        self.MSCA_combined = AttentionModule(dim=in_planes * 4)  # 最终融合维度

        # 最终卷积
        self.finalconv = nn.Conv2d(in_channels=in_planes * 4, out_channels=in_planes, kernel_size=1)

    def forward(self, x):
        # DCT 变换
        LL, LH, HL, HH = self.DCT(x)

        # CBAM 加权
        LH = self.CBAM_LH(LH)
        HL = self.CBAM_HL(HL)
        HH = self.CBAM_HH(HH)
        LL = self.CBAM_LL(LL)

        # 高频特征融合
        high_freq_fusion_input = torch.cat([LH, HL, HH], dim=1)
        high_freq_fusion = self.MSCA_high_freq(high_freq_fusion_input) + high_freq_fusion_input

        # 低频与高频融合
        combined_features_input = torch.cat([LL, high_freq_fusion], dim=1)
        output = self.MSCA_combined(combined_features_input) + combined_features_input

        # 最终卷积
        output = self.finalconv(output)
        return output


if __name__ == '__main__':
    # 测试代码
    x = torch.randn(1, 64, 448, 448).cuda()  # 输入张量 (B, C, H, W)
    module = CombinedModule(img_size=448, in_planes=64, num_freq_masks=4).cuda()  # 初始化模块
    output = module(x)  # 前向传播
    print(output.shape)  # 输出结果
