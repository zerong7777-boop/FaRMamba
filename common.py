import os

import numpy as np
import cv2
import SimpleITK as sitk
import torch
import nibabel as nib
import pydicom
import numpy as np
from PIL import Image

def load_img(filepath, grayscale=False):
    if filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        img = nib.load(filepath)
        img_data = img.get_fdata()
    elif filepath.endswith('.dcm') or filepath.endswith('.IMA'):
        dicom_img = pydicom.dcmread(filepath)
        img_data = dicom_img.pixel_array
    elif filepath.endswith('.png') or filepath.endswith('.jpg') or filepath.endswith('.jpeg'):
        img = Image.open(filepath)
        img_data = np.array(img)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    # 如果需要将图像转换为灰度图像
    if grayscale:
        if img_data.ndim == 3:
            img_data = img_data.mean(axis=2)  # 平均化颜色通道
        elif img_data.ndim == 2:
            img_data = np.expand_dims(img_data, axis=2)  # 确保只有一个通道
        else:
            raise ValueError(f"Unexpected imagesTr dimensions: {img_data.ndim}")

    return img_data


def get_image_dimensions(data):
    """
    获取医学图像的维度和深度信息。

    参数:
    data (numpy.ndarray): 医学图像数据的 NumPy 数组。

    返回:
    tuple: 包含图像尺寸 (宽度, 高度, 深度) 的元组。
    """
    try:
        # 获取图像的维度
        dimensions = data.shape

        # 计算宽度、高度和深度
        width = dimensions[0] if len(dimensions) > 0 else None
        height = dimensions[1] if len(dimensions) > 1 else None
        depth = dimensions[2] if len(dimensions) > 2 else None

        return width, height, depth

    except Exception as e:
        print(f"Error retrieving imagesTr dimensions: {e}")
        return None, None, None

def save_img(img, filepath):
    try:
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)

        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)

        if not filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError(f"Unsupported file format: {filepath}")

        cv2.imwrite(filepath, img)
        print(f"成功保存图像：{filepath}")
    except Exception as e:
        print(f"img的形状：{get_image_dimensions(img)}")
        print(f"保存图像时出错：{e}")


def load_itk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def save_itk(data, path):
    sitk.WriteImage(sitk.GetImageFromArray(data), path)


def tensor2img(tensor, min_max=(0, 1)):
    """
    将 PyTorch 张量转换为图像 NumPy 数组。

    参数：
        tensor (torch.Tensor): 输入的 PyTorch 张量。
        min_max (tuple): 张量的最小和最大值，用于归一化。

    返回：
        numpy.ndarray: 转换后的图像。
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.squeeze().cpu()  # 移除单一维度并转移到 CPU
        img_np = tensor.numpy()  # 转换为 NumPy 数组
    else:
        raise ValueError("输入必须是 PyTorch 张量")

    # 归一化到 [0, 255]
    min_val, max_val = min_max
    img_np = np.clip(img_np, min_val, max_val)  # 确保数值在 min_max 范围内
    img_np = (img_np - min_val) / (max_val - min_val) * 255

    # 转换为 uint8 类型
    img_np = img_np.astype(np.uint8)

    return img_np


def img2tensor(img_arr):
    if img_arr.dtype == np.uint16:
        img_arr = img_arr.astype(np.float32) / 65535.0

    img_tensor = torch.from_numpy(img_arr).float()

    # Add channel dimension if the tensor is 2D (H, W)
    if img_tensor.ndim == 2:
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension (C, H, W)

    # Ensure tensor has 3 dimensions (C, H, W) or 4 dimensions (B, C, H, W)
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.permute(0,3,1,2)
    # Permute if tensor has 4 dimensions (B, C, H, W)
    if img_tensor.ndim == 4:
        # img_tensor = img_tensor.permute(0, 3, 1, 2)  # Convert to BCHW format
        pass
    return img_tensor

def calculate_psnr(img1, img2, crop_border, input_order='HWC', test_y_channel=True):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an imagesTr. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """
    """计算PSNR（峰值信噪比）

    参考：https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    参数:
        img1 (ndarray): 图像范围 [0, 255].
        img2 (ndarray): 图像范围 [0, 255].
        crop_border (int): 图像每边裁剪的像素，这些像素不参与PSNR计算。
        input_order (str): 输入顺序是 'HWC' 还是 'CHW'. 默认: 'HWC'.
        test_y_channel (bool): 是否在YCbCr的Y通道上测试。默认: False.

    返回:
        float: psnr结果.
    """
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    """计算单通道图像的SSIM（结构相似性）

    它由func:`calculate_ssim`调用。

    参数:
        img1 (ndarray): 图像范围 [0, 255]，顺序 'HWC'.
        img2 (ndarray): 图像范围 [0, 255]，顺序 'HWC'.

    返回:
        float: ssim结果.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, crop_border, input_order='HWC', test_y_channel=True):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an imagesTr. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """
    """计算SSIM（结构相似性）

       参考:
       图像质量评估: 从误差可见性到结构相似性

       结果与官方发布的MATLAB代码相同
       https://ece.uwaterloo.ca/~z70wang/research/ssim/。

       对于三通道图像，SSIM对每个通道进行计算，然后取平均值。

       参数:
           img1 (ndarray): 图像范围 [0, 255].
           img2 (ndarray): 图像范围 [0, 255].
           crop_border (int): 图像每边裁剪的像素，这些像素不参与SSIM计算。
           input_order (str): 输入顺序是 'HWC' 还是 'CHW'. 默认: 'HWC'.
           test_y_channel (bool): 是否在YCbCr的Y通道上测试。默认: False.

       返回:
           float: ssim结果.
       """
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def _blocking_effect_factor(im):
    block_size = 8

    block_horizontal_positions = torch.arange(7, im.shape[3] - 1, 8)
    block_vertical_positions = torch.arange(7, im.shape[2] - 1, 8)

    horizontal_block_difference = (
                (im[:, :, :, block_horizontal_positions] - im[:, :, :, block_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_block_difference = (
                (im[:, :, block_vertical_positions, :] - im[:, :, block_vertical_positions + 1, :]) ** 2).sum(3).sum(
        2).sum(1)

    nonblock_horizontal_positions = np.setdiff1d(torch.arange(0, im.shape[3] - 1), block_horizontal_positions)
    nonblock_vertical_positions = np.setdiff1d(torch.arange(0, im.shape[2] - 1), block_vertical_positions)

    horizontal_nonblock_difference = (
                (im[:, :, :, nonblock_horizontal_positions] - im[:, :, :, nonblock_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_nonblock_difference = (
                (im[:, :, nonblock_vertical_positions, :] - im[:, :, nonblock_vertical_positions + 1, :]) ** 2).sum(
        3).sum(2).sum(1)

    n_boundary_horiz = im.shape[2] * (im.shape[3] // block_size - 1)
    n_boundary_vert = im.shape[3] * (im.shape[2] // block_size - 1)
    boundary_difference = (horizontal_block_difference + vertical_block_difference) / (
                n_boundary_horiz + n_boundary_vert)

    n_nonboundary_horiz = im.shape[2] * (im.shape[3] - 1) - n_boundary_horiz
    n_nonboundary_vert = im.shape[3] * (im.shape[2] - 1) - n_boundary_vert
    nonboundary_difference = (horizontal_nonblock_difference + vertical_nonblock_difference) / (
                n_nonboundary_horiz + n_nonboundary_vert)

    scaler = np.log2(block_size) / np.log2(min([im.shape[2], im.shape[3]]))
    bef = scaler * (boundary_difference - nonboundary_difference)

    bef[boundary_difference <= nonboundary_difference] = 0
    return bef


def calculate_psnrb(img1, img2, crop_border, input_order='HWC', test_y_channel=True):
    """Calculate PSNR-B (Peak Signal-to-Noise Ratio).

    Ref: Quality assessment of deblocked images, for JPEG imagesTr deblocking evaluation
    # https://gitlab.com/Queuecumber/quantization-guided-ac/-/blob/master/metrics/psnrb.py

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an imagesTr. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """
    """评估阻塞效应因子（BEF）。

    参考:
    R. A. Ulichney. The void-and-cluster method for dither array generation.
    IEEE Transactions on Image Processing, 1993, 39(3): 1361-1376.

    参数:
        im1 (ndarray): 输入图像。
        im2 (ndarray): 参考图像。

    返回:
        float: BEF得分。
    """
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    # follow https://gitlab.com/Queuecumber/quantization-guided-ac/-/blob/master/metrics/psnrb.py
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) / 255.
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) / 255.

    total = 0
    for c in range(img1.shape[1]):
        mse = torch.nn.functional.mse_loss(img1[:, c:c + 1, :, :], img2[:, c:c + 1, :, :], reduction='none')
        bef = _blocking_effect_factor(img1[:, c:c + 1, :, :])

        mse = mse.view(mse.shape[0], -1).mean(1)
        total += 10 * torch.log10(1 / (mse + bef))

    return float(total) / img1.shape[1]


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input imagesTr.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input imagesTr shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered imagesTr.
    """
    """重新排序图像为 'HWC' 顺序。

    如果输入顺序是 (h, w)，则返回 (h, w, 1)；
    如果输入顺序是 (c, h, w)，则返回 (h, w, c)；
    如果输入顺序是 (h, w, c)，则按原样返回。

    参数：
        img (ndarray)：输入图像。
        input_order (str)：输入顺序的类型，可以是 'HWC' 或 'CHW'。
            如果输入图像的形状是 (h, w)，则 input_order 不会有影响。默认值：'HWC'。

    返回：
        ndarray：重新排序后的图像。
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    """转换为 YCbCr 色彩空间中的 Y 通道。

    参数：
        img (ndarray)：图像，像素值范围为 [0, 255]。

    返回：
        ndarray：Y 通道图像，像素值范围为 [0, 255]（浮点型），没有四舍五入。
    """

    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def _convert_input_type_range(img):
    """Convert the type and range of the input imagesTr.

    It converts the input imagesTr to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input imagesTr in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input imagesTr. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted imagesTr with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the imagesTr according to dst_type.

    It converts the imagesTr to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the imagesTr to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The imagesTr to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the imagesTr to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the imagesTr to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted imagesTr with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR imagesTr to YCbCr imagesTr.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input imagesTr. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr imagesTr. The output imagesTr has the same type
            and range as input imagesTr.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img