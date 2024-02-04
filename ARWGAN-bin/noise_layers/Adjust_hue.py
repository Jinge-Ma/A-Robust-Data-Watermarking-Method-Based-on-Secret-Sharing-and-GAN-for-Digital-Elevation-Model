import torch.nn as nn
import torch
# from kornia.color.adjust import AdjustHue,AdjustSaturation,AdjustContrast,AdjustBrightness,AdjustGamma
from kornia.enhance import adjust_hue, adjust_saturation, adjust_contrast, adjust_brightness, adjust_gamma
import math
from torchvision.transforms import ToPILImage
class Adjust_hue(nn.Module):
    def __init__(self,factor):
        super(Adjust_hue, self).__init__()
        self.factor=factor

    def forward(self, noised_and_cover):
        encoded=((noised_and_cover[0]).clone())
        encoded=adjust_saturation(saturation_factor=self.factor)(encoded)
        noised_and_cover[0]=(encoded)
        return noised_and_cover