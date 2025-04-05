import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import Linear

import pytorch_iou
from models.SSJE import SSJE

RESFEATS = {
    "mobilenet": [16, 24, 32, 96, 320]

}

RESHAPE = {
    "mobilenet": [128, 64, 32, 16, 8]
}

CE = torch.nn.BCELoss().cuda()
iou_loss = pytorch_iou.IOU(size_average=True)


def softmax_one(x, dim=None):
    # subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    # compute exponentials
    exp_x = torch.exp(x)
    # compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class CSAB(nn.Module):
    def __init__(self, config, q_layer_index, k_indexes=None):
        super(CSAB, self).__init__()
        if k_indexes is None:
            k_indexes = [1, 2]
        self.config = config
        self.q_index = q_layer_index
        self.k_indexes = [i + q_layer_index for i in k_indexes]
        self.num_attention_heads = len(self.k_indexes)

        self.query = Linear(RESFEATS[config.backbone][q_layer_index], config.hidden_dim)
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()
        for i in range(0, self.num_attention_heads):
            self.key.append(Linear(RESFEATS[config.backbone][self.k_indexes[i]], config.hidden_dim))
            self.value.append(Linear(RESFEATS[config.backbone][self.k_indexes[i]], config.hidden_dim))

    def forward(self, feature_pyramid):
        B, _, H, W = feature_pyramid[self.q_index].shape
        Q = self.query(rearrange(feature_pyramid[self.q_index], 'b c h w -> b (h w) c'))  # [B, H*W, dim]

        Ks = []  # torch.Size([1, 512, 64, 64]), torch.Size([1, 1024, 32, 32])
        Vs = []
        for i in range(0, self.num_attention_heads):
            # KV shape: torch.Size([1, 512, 128, 128]), torch.Size([1, 1024, 128, 128])
            K = F.interpolate(feature_pyramid[self.k_indexes[i]], size=(H, W), mode='nearest')
            K = self.key[i](rearrange(K, 'b c h w -> b (h w) c'))
            Ks.append(K)

            V = F.interpolate(feature_pyramid[self.k_indexes[i]], size=(H, W), mode='nearest')
            V = self.value[i](rearrange(V, 'b c h w -> b (h w) c'))
            Vs.append(F.pairwise_distance(Q, V, p=2).unsqueeze(-1))  # [Batch, H*W]

        K = torch.cat(Ks, dim=-1)  # [Batch, H*W, heads*dim]
        V = torch.cat(Vs, dim=-1).unsqueeze(-1)  # [Batch, H*W, heads, 1]
        K = K.view(B, -1, self.num_attention_heads, self.config.hidden_dim)  # [Batch, H*W, heads, dim]
        Q = Q.view(B, -1, 1, self.config.hidden_dim)  # [Batch, H*W, 1, dim]

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))  # [Batch, H*W, 1, heads]
        attention_probs = softmax_one(attention_scores, dim=-1)  # [Batch, H*W, 1, heads]
        context_layer = torch.matmul(attention_probs, V).contiguous()  # [Batch, H*W, 1, 1]

        return rearrange(context_layer, 'b (h w) 1 c -> b c h w', h=H)


class HRFM(nn.Module):
    def __init__(
            self,
            hidden_dims=[128, 64, 32, 16],
            input_layers=3,
    ):
        super(HRFM, self).__init__()
        self.up_fuse_layer_0 = UpFuseLayer(input_layers, input_layers, hidden_dims[0], hidden_dims[1])
        self.up_fuse_layer_1 = UpFuseLayer(input_layers, input_layers - 1, hidden_dims[1], hidden_dims[2])
        self.up_fuse_layer_2 = UpFuseLayer(input_layers - 1, 1, hidden_dims[2], hidden_dims[3], relu_flag=False)
        self.out_layer = nn.Conv2d(hidden_dims[-1], 1, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.up_fuse_layer_0(x)
        x = self.up_fuse_layer_1(x)
        x = self.up_fuse_layer_2(x)
        return self.out_layer(torch.relu(x[0]))


class UpFuseLayer(nn.Module):
    def __init__(self, num_branches, out_branches, in_ch, out_ch, norm_layer=None, relu_flag=True):
        super(UpFuseLayer, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        self.relu_flag = relu_flag
        self.out_branches = out_branches

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_branches = num_branches

        self.fuse_layers = self._make_fuse_layers()
        if relu_flag:
            self.relu = nn.ReLU(inplace=True)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []
            for j in range(self.out_branches):
                if j < i - 1:  # 上采样
                    conv3x3s = []
                    for k in range(i - j - 1):
                        if k == i - j - 1:  # 只跨一个分支的下采样
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.out_ch, self.out_ch, 3, 2, 1, bias=False),
                                self.norm_layer(self.out_ch)))
                        else:  # 跨多层的下采样
                            num_outchannels_conv3x3 = self.out_ch
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.in_ch, num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
                else:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.in_ch, self.out_ch, 3, 1, 1, bias=False),
                        self.norm_layer(self.out_ch)))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        x_fuse = []
        for j in range(self.out_branches):
            y = 0
            width_output = x[j].shape[-1] * 2
            height_output = x[j].shape[-2] * 2
            out_shape = (height_output, width_output)
            for i in range(self.num_branches):
                if j <= i - 1:
                    y = y + self.fuse_layers[i][j](x[i])
                else:
                    y = y + F.interpolate(self.fuse_layers[i][j](x[i]), size=out_shape, mode="nearest")

            if self.relu_flag:
                x_fuse.append(self.relu(y))
            else:
                x_fuse.append(y)

        return x_fuse


class DeepSpectralSaliency(nn.Module):
    def __init__(self, config, in_channels=32):
        super(DeepSpectralSaliency, self).__init__()
        self.encoder = SSJE(in_channels)

        self.attention_q_0 = CSAB(config, q_layer_index=0, k_indexes=[1, 2, 3, 4])
        self.attention_q_1 = CSAB(config, q_layer_index=1, k_indexes=[1, 2, 3])
        self.attention_q_2 = CSAB(config, q_layer_index=2, k_indexes=[1, 2])

        self.head = HRFM(hidden_dims=[1, 32, 24, 16], input_layers=3)

        self.pyramid_last = nn.Sequential(
            nn.Conv2d(RESFEATS[config.backbone][-1], 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, gt=None):
        feature_pyramid = self.encoder(x)

        smap_0 = self.attention_q_0(feature_pyramid)
        smap_1 = self.attention_q_1(feature_pyramid)
        smap_2 = self.attention_q_2(feature_pyramid)

        saliency_map = self.head([smap_2, smap_1, smap_0])

        if self.training:
            pyramid_output = self.pyramid_last(feature_pyramid[-1])
            sal_loss = self.cal_loss(saliency_map, gt) + self.cal_loss(pyramid_output, gt)
            return saliency_map, sal_loss
        else:
            return saliency_map

    @staticmethod
    def cal_loss(inp, gt):
        inp = torch.sigmoid(inp)
        if inp.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(gt, inp.shape[-2:], mode="nearest")
        loss = CE(inp, gt) + iou_loss(inp, gt)
        return loss


def get_config():
    config = ml_collections.ConfigDict()
    config.hidden_dim = 256
    config.backbone = "mobilenet"
    return config
