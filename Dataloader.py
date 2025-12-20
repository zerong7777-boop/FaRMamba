from common import load_img, img2tensor
import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
from torchvision import transforms

class PairedData(Dataset):
    '''
    - `root`是数据集的根目录。
    - `target`是数据集的子目录，如训练集或验证集。
    - `use_num`控制加载的样本数量。
    '''

    def __init__(self, root, target='train', use_num=-1, mean=[0.5], std=[0.5]):
        super(PairedData, self).__init__()

        # 定义训练和验证数据路径
        image_folder = os.path.join(root, target, "image")
        label_folder = os.path.join(root, target, "label")

        # 存储图像和标签路径
        self.image_path = []
        self.label_path = []

        # 加载图像和标签路径
        for i, name in enumerate(os.listdir(image_folder)):

            image_file = os.path.join(image_folder, name)
            label_file = os.path.join(label_folder, name)
            self.image_path.append(image_file)
            self.label_path.append(label_file)

            # 如果达到指定数量，提前停止
            if use_num > 0 and i == use_num - 1:
                break

        # 数据集长度
        self.length = len(self.image_path)
        self.target = target

        # 定义归一化转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将图片转换为Tensor
            transforms.Normalize(mean=mean, std=std)  # 归一化处理
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 获取图像和标签路径
        image_path = self.image_path[idx]
        label_path = self.label_path[idx]

        # 加载图像和标签
        image = load_img(image_path, grayscale=True)
        label = load_img(label_path, grayscale=True)
        # 应用归一化
        image = self.transform(image)
        # image = image.unsqueeze(0)
        # print(image.shape)
        # 转换为张量并归一化
        # image = img2tensor(image)
        label = img2tensor(label)



        # 如果标签是分类任务，通常不需要归一化，保持标签原样
        # return image, label ,image_path
        return image, label