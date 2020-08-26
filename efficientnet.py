import torch
import torch.nn as nn
import math

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEblock(nn.Module):
    def __init__(self, ch_in, ch_sq):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_in, ch_sq, 1),
            Swish(),
            nn.Conv2d(ch_sq, ch_in, 1)
        )
        self.se.apply(weights_init)

    def forward(self, x):
        return x * torch.sigmoid(self.se(x))

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class ConvBN(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride = 1, padding = 0, groups = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, groups = groups, bias = False),
            nn.BatchNorm2d(ch_out)
        )
        self.layers.apply(weights_init)

    def forward(self, x):
        return self.layers(x)

class DropConnect(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.training:
            keep_rate = 1.0 - self.drop_rate
            r = torch.rand([x.size(0), 1, 1, 1], dtype = x.dtype).to(x.device)
            r += keep_rate
            mask = r.floor()
            return x.div(keep_rate) * mask
        else:
            return x

class BMConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out,
                expand_ratio, stride, kernel_size, 
                reduction_ratio = 4, drop_connect_rate = 0.2):
        super().__init__()
        self.use_residual = (ch_in == ch_out) & stride == 1
        ch_med = int(ch_in * expand_ratio)
        ch_seq = max(1, ch_in//reduction_ratio)

        if expand_ratio != 1:
            layers = [ConvBN(ch_in, ch_med, 1), Swish()]
        else:
            layers = []

        layers.extend([
            ConvBN(ch_med, ch_med, kernel_size, stride = stride,
            padding = (kernel_size - 1)//2, groups = ch_med),
            Swish(),
            SEblock(ch_med, ch_seq),
            ConvBN(ch_med, ch_out, 1)
        ])
        if self.use_residual:
            self.drop_connect = DropConnect(drop_connect_rate)

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.drop_connect(self.layers(x))
        else:
            return self.layers(x)

class EfficientNet(nn.Module):
    def __init__(self, width_mult = 1.0, depth_mult = 1.0,
                resolution = False, dropout_rate = 0.2,
                input_ch = 3, num_classes = 3):
        super().__init__()

        settings = [
            [1,16,1,1,3],
            [6,24,2,2,3],
            [6,40,2,2,5],
            [6,80,3,2,3],
            [6,112,3,1,5],
            [6,192,4,2,5],
            [6,320,1,1,3]
        ]
        ch_out = int(math.ceil(32*width_mult))
        features = [nn.AdaptiveAvgPool2d(int(224*resolution))] if resolution else []
        features.extend([ConvBN(input_ch, ch_out, 3, stride = 2), Swish()])

        ch_in = ch_out
        for t, c, n, s, k in settings:
            ch_out = int(math.ceil(c*width_mult))
            repeats = int(math.ceil(n*depth_mult))
            for i in range(repeats):
                stride = s if i == 0 else 1
                features.extend([BMConvBlock(ch_in, ch_out, t, stride, k)])
                ch_in = ch_out
        ch_last = int(math.ceil(1280 * width_mult))
        features.extend([ConvBN(ch_in, ch_last, 1), Swish()])
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(ch_last, num_classes)
        )

    def forward(self, x):
        f = self.features(x)
        a = self.avgpool(f)
        c = self.classifier(a)
        return c

def _efficientnet(w_mult, d_mult, resolution, drop_rate, input_ch, num_classes = 3):
    model = EfficientNet(w_mult, d_mult, resolution, drop_rate, input_ch, num_classes)
    return model

def efficientnet_b0(input_ch=3, num_classes=3):
    #(w_mult, d_mult, resolution, droprate) = (1.0, 1.0, 224, 0.2)              
    return _efficientnet(1.0, 1.0, None, 0.2, input_ch, num_classes)

def efficientnet_b1(input_ch=3, num_classes=3):
    #(w_mult, d_mult, resolution, droprate) = (1.0, 1.1, 240, 0.2)              
    return _efficientnet(1.2, 1.1, 1.15, 0.2, input_ch, num_classes)

def efficientnet_b2(input_ch=3, num_classes=3):
    #(w_mult, d_mult, resolution, droprate) = (1.1, 1.2, 260, 0.3)              
    return _efficientnet(1.2**2, 1.1**2, 1.15**2, 0.3, input_ch, num_classes)

def efficientnet_b3(input_ch=3, num_classes=3):
    #(w_mult, d_mult, resolution, droprate) = (1.2, 1.4, 300, 0.3)              
    return _efficientnet(1.2**3, 1.1**3, 1.15**3, 0.3, input_ch, num_classes)

def efficientnet_b4(input_ch=3, num_classes=3):
    #(w_mult, d_mult, resolution, droprate) = (1.4, 1.8, 380, 0.4)              
    return _efficientnet(1.2**4, 1.1**4, 1.15**4, 0.4, input_ch, num_classes)

def efficientnet_b5(input_ch=3, num_classes=3):
    #(w_mult, d_mult, resolution, droprate) = (1.6, 2.2, 456, 0.4)              
    return _efficientnet(1.2**5, 1.1**5, 1.15**5, 0.4, input_ch, num_classes)

def efficientnet_b6(input_ch=3, num_classes=3):
    #(w_mult, d_mult, resolution, droprate) = (1.8, 2.6, 528, 0.5)              
    return _efficientnet(1.2**6, 1.1**6, 1.15**6, 0.5, input_ch, num_classes)

def efficientnet_b7(input_ch=3, num_classes=3):
    #(w_mult, d_mult, resolution, droprate) = (2.0, 3.1, 600, 0.5)              
    return _efficientnet(1.2**7, 1.1**7, 1.15**7, 0.5, input_ch, num_classes)