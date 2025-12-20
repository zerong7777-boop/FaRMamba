import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2  # 使用 PyTorch 的 FFT 接口
from .MSCA import AttentionModule  # MSCA 模块
from .CBAM import CBAM  # CBAM 模块


class FFTTransform(nn.Module):
    def __init__(self, img_size, num_freq_masks):
        super(FFTTransform, self).__init__()
        self.img_size = img_size
        self.num_freq_masks = num_freq_masks
        self.freq_masks = self._generate_freq_masks(img_size, num_freq_masks)

    def _generate_freq_masks(self, img_size, num):
        """生成频域掩码"""
        out = []
        mask_none = torch.ones(img_size, img_size)
        for i in reversed(range(num - 1)):
            mask = torch.ones(img_size // (2 ** i), img_size // (2 ** i))
            mask = torch.triu(mask, diagonal=0)
            mask = torch.rot90(mask, k=1, dims=(0, 1))
            mask = F.pad(mask, (0, img_size - mask.shape[1], 0, img_size - mask.shape[0]), mode='constant', value=0)
            out.append(mask)
        out.append(mask_none)  # 添加无掩码
        return out

    def forward(self, x):
        device = x.device
        batch_size, channels, height, width = x.shape

        # 初始化 4 个频域张量
        LL_tensor = torch.zeros(batch_size, channels, height, width, device=device)
        LH_tensor = torch.zeros(batch_size, channels, height, width, device=device)
        HL_tensor = torch.zeros(batch_size, channels, height, width, device=device)
        HH_tensor = torch.zeros(batch_size, channels, height, width, device=device)

        for i in range(batch_size):
            for c in range(channels):
                # 对每个样本的每个通道进行 2D FFT
                fft_result = fft2(x[i, c])  # 结果为复数

                # 使用掩码划分 FFT 频域
                for idx, mask in enumerate(self.freq_masks):
                    mask = mask.to(device)
                    masked_fft = fft_result * mask  # 应用频域掩码
                    ifft_result = ifft2(masked_fft).real  # IDFT 恢复到实数域

                    # 将还原后的结果存储到对应的频域张量中
                    if idx == 0:
                        LH_tensor[i, c] = ifft_result
                    elif idx == 1:
                        HL_tensor[i, c] = ifft_result
                    elif idx == 2:
                        HH_tensor[i, c] = ifft_result
                    else:
                        LL_tensor[i, c] = ifft_result

        return LL_tensor, LH_tensor, HL_tensor, HH_tensor
class CombinedModule(nn.Module):
    def __init__(self, img_size=256, in_planes=64, num_freq_masks=4):
        super(CombinedModule, self).__init__()
        self.img_size = img_size
        self.initial_in_planes = in_planes

        # FFT 变换模块
        self.FFT = FFTTransform(img_size=img_size, num_freq_masks=num_freq_masks)

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
        # FFT 变换
        LL, LH, HL, HH = self.FFT(x)

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
    x = torch.randn(2, 64, 64, 64).cuda()  # 输入张量 (B, C, H, W)
    module = CombinedModule(img_size=64, in_planes=64, num_freq_masks=4).cuda()  # 初始化模块
    output = module(x)  # 前向传播
    print(output.shape)  # 输出结果
