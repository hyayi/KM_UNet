import os

import cv2
import numpy as np
import torch
import torch.utils.data
import nibabel as nib
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class Dataset(torch.utils.data.Dataset):
    def __init__(self,img_ids,img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # 读取图像
        img = nib.load(os.path.join(self.img_dir, img_id + self.img_ext)).get_fdata()[:,:,0]
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 读取掩码
        mask = nib.load(os.path.join(self.mask_dir, img_id + self.mask_ext)).get_fdata()[:,:]
        mask = mask.astype('uint8')

        # 如果使用了数据增强，则应用变换
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # 归一化图像
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)  # 将 HWC 转为 CHW

        # 归一化掩码
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)  # 将 HWC 转为 CHW

        # 将掩码中的值转为 0 和 1
        if mask.max() < 1:
            mask[mask > 0] = 1.0

        # 显式指定数据类型
        img = torch.tensor(img, dtype=torch.float32)  # 转换为张量并指定数据类型
        mask = torch.tensor(mask, dtype=torch.float32)  # 转换为张量并指定数据类型

        return img, mask, {'img_id': img_id}