import torch
import torch.nn as nn
from .slowfast_feature import SlowFastFeatureExtractor


class RGBIRSlowFast(nn.Module):

    def __init__(self,
                 rgb_weight=None,
                 ir_weight=None,
                 device="cuda"):

        super().__init__()

        # RGB SlowFast（冻结）
        self.rgb_extractor = SlowFastFeatureExtractor(
            weight_path=rgb_weight,
            device=device,
            freeze=True
        )

        # IR SlowFast（训练）
        self.ir_extractor = SlowFastFeatureExtractor(
            weight_path=ir_weight,
            device=device,
            freeze=False
        )

    def forward(self, rgb, ir):

        # IR 单通道转3通道
        ir = [x.repeat(1,3,1,1,1) if x.shape[1]==1 else x for x in ir]

        rgb_feat = self.rgb_extractor(rgb)
        ir_feat = self.ir_extractor(ir)

        return rgb_feat, ir_feat