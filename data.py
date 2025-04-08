from pathlib import Path
import math
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from typing import Tuple, Union

# 配置参数
SCALE_FACTOR = 4  # 粗细分辨率比例（4km -> 1km）
COARSE_PREFIX = '4km'
FINE_PREFIX = '1km'
REF_PREFIXES = ['01', '02']  # 参考影像前缀
PRE_PREFIX = '00'  # 预测影像前缀


class LSTDataset(Dataset):
    """地表温度时空融合数据集"""

    def __init__(self,
                 root_dir: Union[str, Path],
                 image_size: Tuple[int, int],
                 patch_size: Tuple[int, int],
                 n_refs: int = 1,
                 stride: Tuple[int, int] = None):
        """
        Args:
            root_dir (str|Path): 数据集根目录，包含按日期组织的子目录
            image_size (tuple): 原始影像尺寸 (height, width)
            patch_size (tuple): 裁剪块尺寸 (h, w)
            n_refs (int): 参考影像数量（1或2）
            stride (tuple): 滑动步长，默认等于patch_size
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.n_refs = n_refs

        # 获取所有日期子目录
        self.date_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        # 计算分块参数
        self.n_patches_x = (image_size[0] - patch_size[0]) // self.stride[0] + 1
        self.n_patches_y = (image_size[1] - patch_size[1]) // self.stride[1] + 1
        self.total_patches = len(self.date_dirs) * self.n_patches_x * self.n_patches_y

    def __len__(self):
        return self.total_patches

    def _get_patch_coords(self, index: int) -> Tuple[int, int, int]:
        """将全局索引映射到具体子目录和裁剪位置"""
        dir_idx = index // (self.n_patches_x * self.n_patches_y)
        residual = index % (self.n_patches_x * self.n_patches_y)
        x = (residual % self.n_patches_x) * self.stride[0]
        y = (residual // self.n_patches_x) * self.stride[1]
        return dir_idx, x, y

    def _load_image_pair(self, date_dir: Path) -> list:
        """加载指定日期的影像对"""
        paths = []
        # 获取参考影像
        for i in range(self.n_refs):
            paths.extend([
                date_dir / f"{REF_PREFIXES[i]}_{COARSE_PREFIX}.tif",
                date_dir / f"{REF_PREFIXES[i]}_{FINE_PREFIX}.tif"
            ])
        # 添加预测影像
        paths.extend([
            date_dir / f"{PRE_PREFIX}_{COARSE_PREFIX}.tif",
            date_dir / f"{PRE_PREFIX}_{FINE_PREFIX}.tif"
        ])

        images = []
        for p in paths:
            with rasterio.open(str(p)) as src:
                img = src.read().astype(np.float32)  # (C, H, W)
                img[img < 0] = 0  # 处理无效值
                images.append(img)
        return images

    def __getitem__(self, index: int) -> dict:
        # 获取裁剪坐标
        dir_idx, x, y = self._get_patch_coords(index)
        date_dir = self.date_dirs[dir_idx]

        # 加载完整影像对
        full_images = self._load_image_pair(date_dir)

        # 提取patch并转换格式
        patches = {}
        for i, img in enumerate(full_images):
            # 确定当前影像分辨率
            is_coarse = (i % 2 == 0) if i < 2 * self.n_refs else (i == 2 * self.n_refs)
            scale = 1 if not is_coarse else SCALE_FACTOR

            # 计算实际裁剪范围
            h_start = x * scale
            w_start = y * scale
            h_end = h_start + self.patch_size[0] * scale
            w_end = w_start + self.patch_size[1] * scale

            # 提取并转换数据
            patch = img[:, h_start:h_end, w_start:w_end]
            tensor_patch = torch.from_numpy(patch).float() * 0.0001

            # 组织数据结构
            if i < 2 * self.n_refs:
                key = f"ref_{i // 2}_coarse" if is_coarse else f"ref_{i // 2}_fine"
            else:
                key = "pred_coarse" if is_coarse else "pred_fine"
            patches[key] = tensor_patch

        return patches