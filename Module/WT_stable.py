import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from .MSCA import AttentionModule  # MSCA 模块
from .CBAM import CBAM  # CBAM 模块

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db1'):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        device = x.device
        batch_size, channels, height, width = x.shape

        # 初始化输出张量
        LL_tensor = torch.zeros(batch_size, channels, height // 2, width // 2, device=device)
        LH_tensor = torch.zeros(batch_size, channels, height // 2, width // 2, device=device)
        HL_tensor = torch.zeros(batch_size, channels, height // 2, width // 2, device=device)
        HH_tensor = torch.zeros(batch_size, channels, height // 2, width // 2, device=device)

        for i in range(batch_size):
            for c in range(channels):
                # 对每个样本的每个通道进行小波变换
                coeffs = pywt.dwt2(x[i, c].detach().cpu().numpy(), self.wavelet)
                LL, (LH, HL, HH) = coeffs

                # 将结果存储到预先初始化的张量中
                LL_tensor[i, c] = torch.tensor(LL, device=device)
                LH_tensor[i, c] = torch.tensor(LH, device=device)
                HL_tensor[i, c] = torch.tensor(HL, device=device)
                HH_tensor[i, c] = torch.tensor(HH, device=device)

        # 上采样恢复到原始尺寸
        LL_tensor = F.interpolate(LL_tensor, size=(height, width), mode='bilinear', align_corners=False)
        LH_tensor = F.interpolate(LH_tensor, size=(height, width), mode='bilinear', align_corners=False)
        HL_tensor = F.interpolate(HL_tensor, size=(height, width), mode='bilinear', align_corners=False)
        HH_tensor = F.interpolate(HH_tensor, size=(height, width), mode='bilinear', align_corners=False)

        return LL_tensor, LH_tensor, HL_tensor, HH_tensor


class CombinedModule(nn.Module):
    def __init__(self, in_planes=64):
        super(CombinedModule, self).__init__()
        self.initial_in_planes = in_planes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 在 __init__ 中初始化子模块
        msca_dim_high = in_planes * 3
        msca_dim_combined = in_planes * 4
        out_planes = in_planes

        self.WT = WaveletTransform(wavelet='db1')
        self.CBAM_LH = CBAM(in_planes=in_planes)
        self.CBAM_HL = CBAM(in_planes=in_planes)
        self.CBAM_HH = CBAM(in_planes=in_planes)
        self.CBAM_LL = CBAM(in_planes=in_planes)
        self.MSCA_high_freq = AttentionModule(dim=msca_dim_high)
        self.MSCA_combined = AttentionModule(dim=msca_dim_combined)
        self.finalconv = nn.Conv2d(in_channels=msca_dim_combined, out_channels=out_planes, kernel_size=1)

        # 将所有子模块移动到设备
        # 不建议在 __init__ 中调用 self.to(device)
        # 建议在模型实例化后，在外部调用 model.to(device)

    def forward(self, x):
        # 确保输入在正确的设备上
        # x = x.to(self.device)  # 如果需要

        # 小波变换
        LL, LH, HL, HH = self.WT(x)

        # 高频特征处理
        LH = self.CBAM_LH(LH)
        HL = self.CBAM_HL(HL)
        HH = self.CBAM_HH(HH)

        # 高频特征融合
        high_freq_fusion_input = torch.cat([LH, HL, HH], dim=1)
        high_freq_fusion = self.MSCA_high_freq(high_freq_fusion_input) + high_freq_fusion_input

        # 低频特征处理
        LL = self.CBAM_LL(LL)

        # 低频和高频特征融合
        combined_features_input = torch.cat([LL, high_freq_fusion], dim=1)
        output = self.MSCA_combined(combined_features_input) + combined_features_input
        output = self.finalconv(output)
        return output


if __name__ == '__main__':
    x = torch.randn(43, 64, 64, 64).cuda()  # 确保输入张量在正确的设备上
    module = CombinedModule(in_planes=64).cuda()  # 将模型移动到设备
    output = module(x)
    print(output.shape)
