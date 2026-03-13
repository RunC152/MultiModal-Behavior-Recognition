import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    """
    输入: fused 特征 [B, 2H] 
    输出: 分类 logits [B, num_classes]
    """

    def __init__(self, input_dim, num_classes, hidden_dim=None, dropout=0.5):
        """
        input_dim: fused 特征维度
        num_classes: 分类类别数
        hidden_dim: 可选中间隐藏层
        """
        super().__init__()
        if hidden_dim is None:
            # 直接 Linear 输出
            self.fc = nn.Linear(input_dim, num_classes)
            self.dropout = None
        else:
            # 两层 MLP
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x):
        """
        x: [B, input_dim]
        """
        return self.fc(x)