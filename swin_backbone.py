from timm import create_model
import torch.nn as nn

class SwinBackbone(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True, features_only=True):
        super().__init__()
        self.model = create_model(model_name, pretrained=pretrained, features_only=features_only)
    
    def forward(self, x):
        # Returns a list of feature maps from different stages
        features = self.model(x)
        return features
