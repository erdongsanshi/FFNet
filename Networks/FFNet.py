from torchvision import models
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
from Networks.ODConv2d import ODConv2d

__all__ = ['FFNet']

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
 
    def forward(self, x):
        return torch.Tensor.permute(x, self.dims)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)      
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps) 
        x = x.permute(0, 3, 1, 2)      
        return x
    
def conv(in_ch, out_ch, ks, stride):
    
    pad = (ks - 1) // 2
    stage = nn.Sequential(nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ks, stride=stride,
                                       padding=pad, bias=False),
                          LayerNorm2d((out_ch,), eps=1e-06, elementwise_affine=True),
                          nn.GELU(approximate='none'))
    return stage

class ChannelAttention(nn.Module):  
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=self.avg_pool(x)
        avgout = self.shared_MLP(x)
        return self.sigmoid(avgout)
    
class SpatialAttention(nn.Module):  
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.conv1(x)
        return self.sigmoid(x)

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone,self).__init__()
        
        feats =list(convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features.children())
        
        self.stem = nn.Sequential(*feats[0:2])
        self.stage1 = nn.Sequential(*feats[2:4])
        self.stage2 = nn.Sequential(*feats[4:6])
        self.stage3 = nn.Sequential(*feats[6:12])
        
    def forward(self, x):
        x = x.float()
        x = self.stem(x)
        x = self.stage1(x)
        feature1 = x
        x = self.stage2(x)
        feature2 = x
        x = self.stage3(x)
        
        return feature1, feature2, x

class ccsm(nn.Module):
    def __init__(self, channel, channel2, num_filters):
        super(ccsm, self).__init__()
        self.ch_att_s = ChannelAttention(channel)
        self.sa_s = SpatialAttention(7)
        self.conv1 = nn.Sequential(
            ODConv2d(channel, channel, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = channel))
        self.conv2 = nn.Sequential(
            ODConv2d(channel, channel2, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = channel2))
        
        self.conv3 = nn.Sequential(
            ODConv2d(channel2, channel2, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = channel2))
        self.conv4 = nn.Sequential(
            ODConv2d(channel2, num_filters, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = num_filters))
           
    def forward(self, x):
        x = self.ch_att_s(x)*x
        pool1 = x
        x = self.conv1(x)
        x = x + pool1
        x = self.conv2(x)
        pool2 = x
        x = self.conv3(x)
        x = x + pool2
        x = self.conv4(x)
        
        x = self.sa_s(x)*x

        
        return x
    
class Fusion(nn.Module):
    def __init__(self, num_filters1, num_filters2, num_filters3):
        super(Fusion, self).__init__()
        self.upsample_1 = nn.ConvTranspose2d(in_channels=num_filters2, out_channels=num_filters2, kernel_size=4, padding=1, stride=2)
        self.upsample_2 = nn.ConvTranspose2d(in_channels=num_filters3, out_channels=num_filters3, kernel_size=4, padding=0, stride=4)
        self.final = nn.Sequential(
            nn.Conv2d(num_filters1+num_filters2+num_filters3, 1, kernel_size=1, padding=0),
            nn.ReLU(),
        )
        
    def forward(self, x1, x2, x3):
        x2 = self.upsample_1(x2)
        x3 = self.upsample_2(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.final(x)
        
        return x
    
class FFNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_filters = [  16, 32, 64]
        self.backbone = Backbone()
        self.ccsm1 = ccsm(192, 96, num_filters[0])
        self.ccsm2 = ccsm(384, 192, num_filters[1])
        self.ccsm3 = ccsm(768, 384, num_filters[2])
        self.fusion = Fusion(num_filters[0],num_filters[1],num_filters[2])
    def forward(self, x):
        pool1, pool2, pool3 = self.backbone(x)
        
        pool1 = self.ccsm1(pool1)
        pool2 = self.ccsm2(pool2)
        pool3 = self.ccsm3(pool3)
        x = self.fusion(pool1, pool2, pool3)
 
        B, C, H, W = x.size()
        x_sum = x.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        x_normed = x / (x_sum + 1e-6)

        return x, x_normed


if __name__ == '__main__':
    x = torch.rand(size=(16, 3, 512, 512), dtype=torch.float32)
    model = FFNet()
    
    mu, mu_norm = model(x)
    print(mu.size(), mu_norm.size())
    