''' AViT implementation taken from
https://github.com/NVlabs/A-ViT
'''

# --------------------------------------------------------
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2022 paper
# A-ViT: Adaptive Tokens for Efficient Vision Transformer
# Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz,
# and Pavlo Molchanov
# --------------------------------------------------------

import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.FiLM import FiLM_Layer, FiLM_ViT_Layer
from models.DualBN import DualBN2d, DualLN

from timm.models.vision_transformer import _cfg, _init_vit_weights
from timm.models.helpers import named_apply
from timm.models.layers import DropPath, PatchEmbed, Mlp, trunc_normal_
from utils_avit import get_distribution_target

from torch.autograd import Variable

import numpy as np

_logger = logging.getLogger(__name__)


__all__ = [
    'avit_tiny_patch16_32'
]

def avit_tiny_patch16_32(num_classes=10, **kwargs):

    model = ACTVisionTransformer(
        img_size=32, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, 
        qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(num_classes=num_classes, input_size=(3, 32, 32))  # TODO: maybe fix mean and std too

    return model


# Adaptive Vision Transformer
class ACTVisionTransformer(nn.Module):
    """ Vision Transformer with Adaptive Token Capability

    Starting at:
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929

        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877

    Extended to:
        Accomodate adaptive token inference
    """
    

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', FiLM_in_channels=1, use2BN=False, args=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1

        self.use2LN = use2BN
        if self.use2LN:
            norm_layer = partial(DualLN, eps=1e-6)
        else:
            norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block_ACT_OAT(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, args=args, 
                index=i, num_patches=self.patch_embed.num_patches+1, use2LN=use2BN, FiLM_in_channels=FiLM_in_channels)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

        print('\nNow this is an OAT ACT DeiT.\n')
        self.eps = 0.01
        print(f'Setting eps as {self.eps}.')

        print('Now re-initializing the halting network bias')
        for block in self.blocks:
            if args.act_mode == 1:
                # torch.nn.init.constant_(block.act_mlp.fc1.bias.data, -3)
                torch.nn.init.constant_(block.act_mlp.fc2.bias.data, -1. * args.gate_center)

        self.args = args

        print('Now setting up the rho.')
        self.rho = None  # Ponder cost
        self.counter = None  # Keeps track of how many layers are used for each example (for logging)
        self.batch_cnt = 0 # amount of batches seen, mainly for tensorboard

        # for token act part
        self.c_token = None
        self.R_token = None
        self.mask_token = None
        self.rho_token = None
        self.counter_token = None
        self.total_token_cnt = num_patches + self.num_tokens

        if args.distr_prior_alpha >0. :
            self.distr_target = torch.Tensor(get_distribution_target(standardized=True)).cuda()
            self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()


    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


    def forward_features_act_token(self, x, _lambda, idx2BN=None):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # now start the act part
        bs = x.size()[0]  # The batch size

        # this part needs to be modified for higher GPU utilization
        if self.c_token is None or bs != self.c_token.size()[0]:
            self.c_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.R_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.mask_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.rho_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.counter_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())

        c_token = self.c_token.clone()
        R_token = self.R_token.clone()
        mask_token = self.mask_token.clone()
        self.rho_token = self.rho_token.detach() * 0.
        self.counter_token = self.counter_token.detach() * 0 + 1.
        # Will contain the output of this residual layer (weighted sum of outputs of the residual blocks)
        output = None
        # Use out to backbone
        out = x

        if self.args.distr_prior_alpha>0.:
            self.halting_score_layer = []

        for i, l in enumerate(self.blocks):

            # block out all the parts that are not used
            out.data = out.data * mask_token.float().view(bs, self.total_token_cnt, 1)

            # evaluate layer and get halting probability for each sample
            # block_output, h_lst = l.forward_act(out)    # h is a vector of length bs, block_output a 3D tensor
            # block_output, h_lst = l.forward_act(out, 1.-mask_token.float())    # h is a vector of length bs, block_output a 3D tensor
            block_output, h_lst = l.forward_act(out, _lambda, idx2BN, 1.-mask_token.float())    # h is a vector of length bs, block_output a 3D tensor
            # henry: replaced with oated version

            if self.args.distr_prior_alpha>0.:
                self.halting_score_layer.append(torch.mean(h_lst[1][1:]))

            out = block_output.clone()              # Deep copy needed for the next layer

            _, h_token = h_lst # h is layer_halting score, h_token is token halting score, first position discarded

            # here, 1 is remaining, 0 is blocked
            block_output = block_output * mask_token.float().view(bs, self.total_token_cnt, 1)

            # Is this the last layer in the block?
            if i==len(self.blocks)-1:
                h_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())

            # for token part
            c_token = c_token + h_token
            self.rho_token = self.rho_token + mask_token.float()

            # Case 1: threshold reached in this iteration
            # token part
            reached_token = c_token > 1 - self.eps
            reached_token = reached_token.float() * mask_token.float()
            delta1 = block_output * R_token.view(bs, self.total_token_cnt, 1) * reached_token.view(bs, self.total_token_cnt, 1)
            self.rho_token = self.rho_token + R_token * reached_token

            # Case 2: threshold not reached
            # token part
            not_reached_token = c_token < 1 - self.eps
            not_reached_token = not_reached_token.float()
            R_token = R_token - (not_reached_token.float() * h_token)
            delta2 = block_output * h_token.view(bs, self.total_token_cnt, 1) * not_reached_token.view(bs, self.total_token_cnt, 1)

            self.counter_token = self.counter_token + not_reached_token # These data points will need at least one more layer

            # Update the mask
            mask_token = c_token < 1 - self.eps

            if output is None:
                output = delta1 + delta2
            else:
                output = output + (delta1 + delta2)

        if (self.use2LN):
            x = self.norm(output, idx2BN)
        else:
            x = self.norm(output)
        # TODO: maybe film here?

        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]


    def forward_probs(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        out_lst = []
        assert self.dist_token is None

        for i, l in enumerate(self.blocks):
            # evaluate layer and get halting probability for each sample
            out = l.forward(x)    # h is a vector of length bs, block_output a 3D tensor
            tmp_prob = self.head(self.pre_logits(self.norm(out)[:, 0]))
            out_lst.append(tmp_prob)
            x = out

        return out_lst


    def forward(self, x, _lambda, idx2BN=None):
        if self.args.act_mode == 4:
            x = self.forward_features_act_token(x, _lambda, idx2BN)
        else:
            print('Not implemented yet, please specify for token act.')
            exit()

        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        # return x, rho, count # discarded from v1
        return x


class Block_ACT_OAT(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, args=None, index=-1, num_patches=197,
                 use2LN=False, FiLM_in_channels=1):
        super().__init__()
        self.use2LN = use2LN
        
        self.norm1 = norm_layer(dim)
        self.film1 = FiLM_ViT_Layer(embed_dim=dim, in_channels=FiLM_in_channels) 
        self.attn = Masked_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.film2 = FiLM_ViT_Layer(embed_dim=dim, in_channels=FiLM_in_channels) 
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.act_mode = args.act_mode
        assert self.act_mode in {1, 2, 3, 4} #now only support 1-extra mlp, or b-position 0 encoding

        self.index=index
        self.args = args

        if self.act_mode == 4:
            # Apply sigmoid on the mean of all tokens to determine whether to continue
            self.sig = torch.sigmoid
        else:
            print('Not supported yet.')
            exit()

    def forward(self, x, _lambda, idx2BN=None):
        x_res = x
        if (self.use2LN):
            x = self.norm1(x, idx2BN)
        else:
            x = self.norm1(x)
        x = self.film1(x, _lambda)
        
        x = x_res + self.drop_path(self.attn(x))

        x_res = x
        if (self.use2LN):
            x = self.norm2(x, idx2BN)
        else:
            x = self.norm2(x)
        x = self.film2(x, _lambda)

        x = x_res + self.drop_path(self.mlp(x))

        return x


    def forward_act(self, x, _lambda, idx2BN=None, mask=None):

        debug=False
        analyze_delta = True
        bs, token, dim = x.shape
        x_res = x

        if mask is None:
            if (self.use2LN):
                x = self.norm1(x, idx2BN)
            else:
                x = self.norm1(x)
            x = self.film1(x, _lambda)
            
            x = x_res + self.drop_path(self.attn(x))

            x_res = x
            if (self.use2LN):
                x = self.norm2(x, idx2BN)
            else:
                x = self.norm2(x)
            x = self.film2(x, _lambda)

            x = x_res + self.drop_path(self.mlp(x))
        else:
            if (self.use2LN):
                x = self.norm1(x*(1-mask).view(bs, token, 1), idx2BN)*(1-mask).view(bs, token, 1)
            else:
                x = self.norm1(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1)
            x = self.film1(x, _lambda)
            
            x = x_res + self.drop_path(self.attn(x, mask=mask))

            x_res = x
            if (self.use2LN):
                x = self.norm2(x*(1-mask).view(bs, token, 1), idx2BN)*(1-mask).view(bs, token, 1)
            else:
                x = self.norm2(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1)
            x = self.film2(x, _lambda)

            x = x_res + self.drop_path(self.mlp(x))

        if self.act_mode==4:
            gate_scale, gate_center = self.args.gate_scale, self.args.gate_center
            halting_score_token = self.sig(x[:,:,0] * gate_scale - gate_center)
            # initially first position used for layer halting, second for token
            # now discarding position 1
            halting_score = [-1, halting_score_token]
        else:
            print('Not supported yet.')
            exit()

        return x, halting_score


class Masked_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., mask=None, masked_softmax_bias=-1000.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask = mask # this is of shape [batch, token_number], where the token number
                         # dimension is indication of token exec.
                         # 0's are the tokens to continue, 1's are the tokens masked out

        self.masked_softmax_bias = masked_softmax_bias

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # now we need to mask out all the attentions associated with this token
            attn = attn + mask.view(mask.shape[0], 1, 1, mask.shape[1]) * self.masked_softmax_bias
            # this additional bias will make attention associated with this token to be zeroed out
            # this incurs at each head, making sure all embedding sections of other tokens ignore these tokens

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class BasicBlockOAT(nn.Module):

    def __init__(self, in_planes, mid_planes, out_planes, stride=1, use2BN=False, FiLM_in_channels=1):
        super(BasicBlockOAT, self).__init__()
        self.use2BN = use2BN
        if self.use2BN:
            Norm2d = DualBN2d
        else:
            Norm2d = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = Norm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = Norm2d(out_planes)

        if stride != 1 or in_planes != out_planes:
            self.mismatch = True
            self.conv_sc = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            self.bn_sc = Norm2d(out_planes)
        else:
            self.mismatch = False
        
        self.film1 = FiLM_Layer(channels=mid_planes, in_channels=FiLM_in_channels) 
        self.film2 = FiLM_Layer(channels=out_planes, in_channels=FiLM_in_channels)

    def forward(self, x, _lambda, idx2BN=None):
        out = self.conv1(x)
        if self.use2BN:
            out = self.bn1(out, idx2BN)
        else:
            out = self.bn1(out)
        out = self.film1(out, _lambda)
        out = F.relu(out)
        out = self.conv2(out)
        if self.use2BN:
            out = self.bn2(out, idx2BN)
        else:
            out = self.bn2(out)
        out = self.film2(out, _lambda)
        if self.mismatch:
            if self.use2BN: 
                out += self.bn_sc(self.conv_sc(x), idx2BN)
            else:
                out += self.bn_sc(self.conv_sc(x))
        else:
            out += x
        out = F.relu(out)
        # print(out.size())
        return out



class ResNet34OAT(nn.Module):
    '''
    GFLOPS: 1.1837, model size: 31.4040MB
    '''
    def __init__(self, num_classes=10, FiLM_in_channels=1, use2BN=False):
        super(ResNet34OAT, self).__init__()
        self.use2BN = use2BN

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use2BN:
            self.bn1 = DualBN2d(64)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.film1 = FiLM_Layer(channels=64, in_channels=FiLM_in_channels)
        self.bundle1 = nn.ModuleList([
            BasicBlockOAT(64, 64, 64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(64, 64, 64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(64, 64, 64, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle2 = nn.ModuleList([
            BasicBlockOAT(64, 128, 128, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(128, 128, 128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(128, 128, 128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(128, 128, 128, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle3 = nn.ModuleList([
            BasicBlockOAT(128, 256, 256, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(256, 256, 256, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.bundle4 = nn.ModuleList([
            BasicBlockOAT(256, 512, 512, stride=2, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(512, 512, 512, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
            BasicBlockOAT(512, 512, 512, stride=1, use2BN=use2BN, FiLM_in_channels=FiLM_in_channels),
        ])
        self.linear = nn.Linear(512, num_classes)
        self.bundles = [self.bundle1, self.bundle2, self.bundle3, self.bundle4]

    def forward(self, x, _lambda, idx2BN=None):
        out = self.conv1(x)
        if self.use2BN:
            out = self.bn1(out, idx2BN)
        else:
            out = self.bn1(out)
        out = self.film1(out, _lambda)
        out = F.relu(out)
        for bundle in self.bundles:
            for block in bundle:
                out = block(out, _lambda, idx2BN)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    from thop import profile
    net = ResNet34OAT()
    x = torch.randn(1,3,32,32)
    _lambda = torch.ones(1,1)
    flops, params = profile(net, inputs=(x, _lambda))
    y = net(x, _lambda)
    print(y.size())
    print('GFLOPS: %.4f, model size: %.4fMB' % (flops/1e9, params/1e6))