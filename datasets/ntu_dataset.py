import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
from pytorchvideo.data import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda

class NTUDataset(Dataset):
    """
    多模态视频数据集，同时加载 RGB 和 IR 视频
    适配 NTU RGB+D 格式：S001C003P008R002A048_rgb.avi 和 S001C003P008R002A048_ir.avi
    """
    def __init__(
        self,
        rgb_dir,
        ir_dir,
        transform_rgb=None,
        transform_ir=None,
        num_frames=32,
    ):
        """
        参数说明：
        - rgb_dir: RGB 视频文件夹路径
        - ir_dir: IR 红外视频文件夹路径
        - transform_rgb: RGB 视频预处理变换
        - transform_ir: IR 视频预处理变换
        - num_frames: 每个视频采样的帧数
        """
        self.rgb_dir = Path(rgb_dir)
        self.ir_dir = Path(ir_dir)
        
        # 获取所有 RGB 和 IR 视频文件
        self.rgb_files = sorted(list(self.rgb_dir.glob("*_rgb.avi")))
        self.ir_files = sorted(list(self.ir_dir.glob("*_ir.avi")))
        
        # 按文件名前缀匹配成对的样本（确保一一对应）
        self.pairs = self._match_pairs()
        
        # 标签映射
        self.label_map = self._build_label_map()
        
        # 预处理变换
        self.transform_rgb = transform_rgb if transform_rgb is not None \
            else self._get_default_transform(num_frames, is_rgb=True)
        self.transform_ir = transform_ir if transform_ir is not None \
            else self._get_default_transform(num_frames, is_rgb=False)

    def _match_pairs(self):
        """按文件名前缀匹配 RGB 和 IR 视频对"""
        pairs = []
        # 用字典存储 IR 视频，方便快速查找
        ir_dict = {f.stem.replace("_ir", ""): f for f in self.ir_files}
        
        for rgb_file in self.rgb_files:
            prefix = rgb_file.stem.replace("_rgb", "")
            if prefix in ir_dict:
                pairs.append((rgb_file, ir_dict[prefix]))
            else:
                print(f"警告：未找到与 {rgb_file.name} 匹配的 IR 视频，已跳过")
        
        if len(pairs) == 0:
            raise ValueError("未找到任何匹配的 RGB-IR 视频对，请检查文件夹路径和文件名格式")
        
        return pairs

    def _build_label_map(self):
        """从文件名解析动作标签（S001C003P008R002A048 -> 48）"""
        label_map = {}
        for idx, (rgb_file, _) in enumerate(self.pairs):
            filename = rgb_file.stem
            action_part = filename.split("_")[0].split("A")[-1]
            label = int(action_part)
            label_map[idx] = label
        return label_map

    def _get_default_transform(self, num_frames, is_rgb=True):
        """默认预处理变换，RGB 和 IR 可使用不同的均值/方差"""
        if is_rgb:
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
        else:
            # IR 是单通道，均值/方差可根据数据统计调整
            mean = [0.5]
            std = [0.5]
        
        return ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                Normalize(mean=mean, std=std),
                ShortSideScale(size=256),
            ])
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_file, ir_file = self.pairs[idx]
        label = self.label_map[idx]
        
        try:
            # 加载 RGB 视频
            rgb_video = EncodedVideo.from_path(str(rgb_file))
            rgb_data = rgb_video.get_clip(start_sec=0, end_sec=None)
            rgb_data = self.transform_rgb(rgb_data)
            rgb_tensor = rgb_data["video"]  # shape: (3, T, H, W)
            
            # 加载 IR 视频
            ir_video = EncodedVideo.from_path(str(ir_file))
            ir_data = ir_video.get_clip(start_sec=0, end_sec=None)
            ir_data = self.transform_ir(ir_data)
            ir_tensor = ir_data["video"]  # shape: (1, T, H, W)
            
            return {
                "rgb": rgb_tensor,
                "ir": ir_tensor,
                "label": torch.tensor(label, dtype=torch.long)
            }
        
        except Exception as e:
            print(f"读取样本 {idx} 失败: {e}")
            # 返回空张量作为占位符
            return {
                "rgb": torch.zeros(3, 32, 256, 256),
                "ir": torch.zeros(1, 32, 256, 256),
                "label": torch.tensor(-1, dtype=torch.long)
            }