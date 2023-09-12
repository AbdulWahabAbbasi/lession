import torch
import torch.nn as nn


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))
        #self.add_module('avgpool', nn.AvgPool2d(2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

##################################################################daptiveFeatureSelectionLayer
# import torch.nn as nn
# class SqueezeAttentionBlock(nn.Module):
#     def __init__(self, ch_in, ch_out, intermediate_channels=None):
#         super(SqueezeAttentionBlock, self).__init__()
#
#         if intermediate_channels is None:
#             intermediate_channels = ch_out // 2  # default to half of ch_out
#
#         self.weight_network = nn.Sequential(
#             nn.Conv2d(ch_in, intermediate_channels, kernel_size=1),
#             nn.BatchNorm2d(intermediate_channels),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(intermediate_channels),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(intermediate_channels, ch_out, kernel_size=1),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(ch_out, ch_out, kernel_size=1),
#             nn.BatchNorm2d(ch_out),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         weights = self.weight_network(x)
#         return x * weights

########################################################################################Squeeze-and-Excitation Network
import torch.nn.functional as F
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(ch_out)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out)
        return out
# ######################################################3##################################ChannelAttentionBlock
# class SqueezeAttentionBlock(nn.Module):
#     def __init__(self, ch_in, ch_out, reduction=16):
#         super(SqueezeAttentionBlock, self).__init__()
#         self.conv = conv_block(ch_in, ch_out)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(ch_in, ch_in // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(ch_in // reduction, ch_in, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x_res = self.conv(x)
#         avg_out = self.fc(self.avg_pool(x_res).view(x_res.size(0), -1)).view(x_res.size(0), x_res.size(1), 1, 1)
#         max_out = self.fc(self.max_pool(x_res).view(x_res.size(0), -1)).view(x_res.size(0), x_res.size(1), 1, 1)
#         out = avg_out + max_out
#         return x_res * out
##################################################################SqueezeAttentionBlock
# class SqueezeAttentionBlock(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(SqueezeAttentionBlock, self).__init__()
#         self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv = conv_block(ch_in, ch_out)
#         self.conv_atten = conv_block(ch_in, ch_out)
#         self.upsample = nn.Upsample(scale_factor=2)
#
#     def forward(self, x):
#         #print("x.shape: ", x.shape)
#         x_res = self.conv(x)
#         #print("x_res.shape: ", x_res.shape)
#         y = self.avg_pool(x)
#         #print("y.shape dopo avg pool: ", y.shape)
#         y = self.conv_atten(y)
#         #print("y.shape dopo conv att:", y.shape)
#         y = self.upsample(y)
#         #print(y.shape, x_res.shape)
#         #print("(y * x_res) + y: ", (y * x_res) + y)
#         return (y * x_res) + y
#
def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

# class WindowAttention(nn.Module):
#     def __init__(self, dim, window_size, heads):
#         super().__init__()
#         self.heads = heads
#         self.scale = window_size ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.attn_drop = nn.Dropout(0.1)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(0.1)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = x.permute(0, 2, 3, 1)
#         qkv = self.qkv(x).reshape(B, H * W, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
#         x = self.proj(x.permute(0, 3, 1, 2))
#         x = self.proj_drop(x)
#         return x
#
# class SwinTransformerBlock(nn.Module):
#     def __init__(self, dim, heads, window_size=7):
#         super().__init__()
#         self.norm1 = nn.BatchNorm2d(dim)
#         self.attn = WindowAttention(dim, window_size, heads)
#         self.norm2 = nn.BatchNorm2d(dim)
#         self.mlp_head = nn.Linear(dim * window_size * window_size, dim)  # account for H, W dims
#         self.mlp_tail = nn.Linear(dim, dim)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = x + self.attn(self.norm1(x))
#         x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
#         x_mlp = self.mlp_tail(self.mlp_head(x_flat))
#         x_mlp = x_mlp.view(B, H, W, C).permute(0, 3, 1, 2)
#         x = x + self.norm2(x_mlp)
#         return x
#
# class SqueezeAttentionBlock(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(SqueezeAttentionBlock, self).__init__()
#         self.swin_block = SwinTransformerBlock(ch_in, heads=4)  # Adjust heads as required
#
#     def forward(self, x):
#         return self.swin_block(x)
#
# # Rest of your classes and functions remain unchanged...
#
# def center_crop(layer, max_height, max_width):
#     _, _, h, w = layer.size()
#     xy1 = (w - max_width) // 2
#     xy2 = (h - max_height) // 2
#     return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

