import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import copy

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        
        self.bn_a   = nn.BatchNorm2d(in_channels,affine=True)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)
        self.bn     = nn.BatchNorm2d(out_channels,affine=True)

    def forward(self, x):
        x = self.bn_a(self.conv_a(x))
        x = self.bn(self.conv(x))
        return x

class LightCNN9_feature(nn.Module):
    def __init__(self,feature_dim,load_path=None,gabor_out=4):
        super(LightCNN9_feature,self).__init__()   
        self.features = nn.Sequential(
            mfm(gabor_out, 48, 5, 1, 2), 
            nn.BatchNorm2d(48,affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(48, 96, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )

        self.fc1 = mfm(8*8*128, feature_dim, type=0)
        if load_path is not None:
            self._param_load(load_path)
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    

def gabor_fn(kernel_size, channel_in, channel_out, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma    # [channel_out]
    sigma_y = sigma.float() / gamma     # element-wize division, [channel_out]
    nstds = 3 
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax+1)
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax+1)
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()   # [channel_out, channelin, kernel, kernel]
    if theta.is_cuda:
        x = x.cuda()
        y = y.cuda()

    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

    gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2)) \
         * torch.cos(2 * math.pi / Lambda.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

    return gb

class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, premodel_path=None):
        super(GaborConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding

        self.Lambda = nn.Parameter(torch.rand(channel_out), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.psi = nn.Parameter(torch.randn(channel_out) * 0.02, requires_grad=True)
        self.sigma = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(channel_out) * 0.0, requires_grad=True)

        self.sigmoid = nn.Sigmoid()
        if premodel_path is not None:
            checkpoint = torch.load(premodel_path, map_location=lambda storage, loc: storage)
            model_dict = self.state_dict()
            pretrain_model = {k: v for k,v in checkpoint['pre_net'].items() if (k in model_dict) & (k in model_dict and (v.size()==model_dict[k].size() and 1 or 0)or 0)}
            print('GaborNet:   loading {} parameters'.format(len(pretrain_model.keys())))
            model_dict.update(pretrain_model)
            self.load_state_dict(model_dict)

    def forward(self, x):
        theta = self.sigmoid(self.theta) * math.pi * 2.0
        gamma = 1.0 + (self.gamma * 0.5)
        sigma = 0.1 + (self.sigmoid(self.sigma) * 0.4)
        Lambda = 0.001 + (self.sigmoid(self.Lambda) * 0.999)
        psi = self.psi

        kernel = gabor_fn(self.kernel_size, self.channel_in, self.channel_out, sigma, theta, Lambda, psi, gamma)
        kernel = kernel.float()  
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out

class GaborTridentNet(nn.Module):
    def __init__(self, feature_dim):
        super(GaborTridentNet, self).__init__()
        self.prenet_NIR, self.prenet_VIS = self._copy_net_gen(GaborConv2d(1,4,7,stride=1, padding=3),num=2)
        featureNet = nn.Sequential(
            mfm(4, 48, 5, 1, 2), 
            nn.BatchNorm2d(48,affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(48, 96, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.BatchNorm2d(128, eps=2e-5, affine=False),
            # nn.Dropout(p=0.4),
            Flatten(),
            mfm(8*8*128, feature_dim, type=0),
            nn.BatchNorm1d(feature_dim, eps=2e-5))
        # del featureNet[11]

        self.feat_comm, self.feat_nir, self.feat_vis = self._copy_net_gen(featureNet,num=3)

        self.residual_weight = nn.Parameter(torch.randn(1))
        
    def _copy_net_gen(self, net, num):
        copied_net = []
        for _ in range(num):
            copied_net.append(copy.deepcopy(net))
        return copied_net

    def forward(self, x_nir, x_vis, labels):   
        x_nir, x_vis = self.prenet_NIR(x_nir), self.prenet_VIS(x_vis)
        nir_comm = self.feat_comm(x_nir)
        vis_comm = self.feat_comm(x_vis)
        nir_spec = self.feat_nir(x_nir)
        vis_spec = self.feat_vis(x_vis)
        x_nir = nir_comm * (1-torch.sigmoid(self.residual_weight)) + nir_spec * torch.sigmoid(self.residual_weight)
        x_vis = vis_comm * (1-torch.sigmoid(self.residual_weight)) + vis_spec * torch.sigmoid(self.residual_weight)
    
        return x_nir, x_vis

def dullightcnn_zoo(feat_dim = 256):
    return GaborTridentNet(feat_dim)
