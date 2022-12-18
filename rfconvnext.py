""" RFConvNeXt
Paper: RF-Next: Efficient Receptive Field Search for Convolutional Neural Networks
    https://arxiv.org/abs/2206.06637
    
Modified from https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/convnext.py
"""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import named_apply, build_model_with_cfg, checkpoint_seq
from timm.models.layers import trunc_normal_, SelectAdaptivePool2d, DropPath, ConvMlp, Mlp, LayerNorm2d,\
    create_conv2d, make_divisible, get_padding
from .rfconv import RFConv2d
import os

__all__ = ['RFConvNeXt']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = dict(
    convnext_tiny=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"),
    convnext_small=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth"),
    convnext_base=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth"),
    convnext_large=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth"),

    # timm specific variants
    convnext_atto=_cfg(url=''),
    convnext_atto_ols=_cfg(url=''),
    convnext_femto=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_femto_d1-d71d5b4c.pth',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    convnext_femto_ols=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_femto_ols_d1-246bf2ed.pth',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    convnext_pico=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_pico_d1-10ad7f0d.pth',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    convnext_pico_ols=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_pico_ols_d1-611f0ca7.pth',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    convnext_nano=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_nano_d1h-7eb4bdea.pth',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    convnext_nano_ols=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_nano_ols_d1h-ae424a9a.pth',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    convnext_tiny_hnf=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_tiny_hnf_a2h-ab7e9df2.pth',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),

    convnext_tiny_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth'),
    convnext_small_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth'),
    convnext_base_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth'),
    convnext_large_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth'),
    convnext_xlarge_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth'),

    convnext_tiny_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_small_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_base_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_large_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_xlarge_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),

    convnext_tiny_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth", num_classes=21841),
    convnext_small_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth", num_classes=21841),
    convnext_base_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth", num_classes=21841),
    convnext_large_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth", num_classes=21841),
    convnext_xlarge_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth", num_classes=21841),
)


default_search_cfg = dict(
    num_branches=3,
    expand_rate=0.5,
    max_dilation=None,
    min_dilation=1,
    init_weight=0.01,
    search_interval=1250,
    max_search_step=0,
)


class RFConvNeXtBlock(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            dim_out=None,
            stride=1,
            dilation=1,
            mlp_ratio=4,
            conv_mlp=False,
            conv_bias=True,
            ls_init_value=1e-6,
            norm_layer=None,
            act_layer=nn.GELU,
            drop_path=0.,
            search_cfgs=default_search_cfg
    ):
        super().__init__()
        dim_out = dim_out or dim
        if not norm_layer:
            norm_layer = partial(LayerNorm2d, eps=1e-6) if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        mlp_layer = ConvMlp if conv_mlp else Mlp
        self.use_conv_mlp = conv_mlp

        # replace dwconv with rfconv
        self.conv_dw = RFConv2d(
            in_channels=dim, 
            out_channels=dim_out, 
            kernel_size=7,
            stride=stride, 
            padding=get_padding(kernel_size=7, stride=stride, dilation=dilation),
            dilation=dilation,
            groups=dim,
            bias=conv_bias,
            **search_cfgs)
        self.norm = norm_layer(dim_out)
        self.mlp = mlp_layer(dim_out, int(mlp_ratio * dim_out), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim_out)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        x = self.drop_path(x) + shortcut
        return x


class RFConvNeXtStage(nn.Module):

    def __init__(
            self,
            in_chs,
            out_chs,
            stride=2,
            depth=2,
            dilation=(1, 1),
            drop_path_rates=None,
            ls_init_value=1.0,
            conv_mlp=False,
            conv_bias=True,
            norm_layer=None,
            norm_layer_cl=None,
            search_cfgs=default_search_cfg
    ):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = 'same' if dilation[1] > 1 else 0  # same padding needed if dilation used
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                create_conv2d(
                    in_chs, out_chs, kernel_size=ds_ks, stride=stride,
                    dilation=dilation[0], padding=pad, bias=conv_bias),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(RFConvNeXtBlock(
                dim=in_chs,
                dim_out=out_chs,
                dilation=dilation[1],
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                norm_layer=norm_layer if conv_mlp else norm_layer_cl,
                search_cfgs=search_cfgs
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class RFConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        rf_mode (str): Training mode for RF-Next. Choose from ['rfsearch', 'rfsingle', 'rfmultiple', 'rfmerge'].
        kernel_cfgs (Dict(str, int)): Kernel size for each RFConv. Example: {"stages.0.blocks.0.conv_dw": 7, "stages.0.blocks.1.conv_dw": 7, ...}.
    """

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            output_stride=32,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            ls_init_value=1e-6,
            stem_type='patch',
            patch_size=4,
            head_init_scale=1.,
            head_norm_first=False,
            conv_mlp=False,
            conv_bias=True,
            norm_layer=None,
            drop_rate=0.,
            drop_path_rate=0.,
            pretrained_weights=None,
            rf_mode='rfsearch',
            kernel_cfgs=None,
            search_cfgs=default_search_cfg
    ):
        super().__init__()
        assert output_stride in (8, 16, 32)
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)
            norm_layer_cl = norm_layer if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        else:
            assert conv_mlp,\
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            norm_layer_cl = norm_layer

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        assert stem_type in ('patch', 'overlap', 'overlap_tiered')
        if stem_type == 'patch':
            # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=conv_bias),
                norm_layer(dims[0])
            )
            stem_stride = patch_size
        else:
            mid_chs = make_divisible(dims[0] // 2) if 'tiered' in stem_type else dims[0]
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, mid_chs, kernel_size=3, stride=2, padding=1, bias=conv_bias),
                nn.Conv2d(mid_chs, dims[0], kernel_size=3, stride=2, padding=1, bias=conv_bias),
                norm_layer(dims[0]),
            )
            stem_stride = 4

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        curr_stride = stem_stride
        dilation = 1
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(RFConvNeXtStage(
                prev_chs,
                out_chs,
                stride=stride,
                dilation=(first_dilation, dilation),
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl,
                search_cfgs=search_cfgs
            ))
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs

        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
        self.norm_pre = norm_layer(self.num_features) if head_norm_first else nn.Identity()
        self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', nn.Identity() if head_norm_first else norm_layer(self.num_features)),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('drop', nn.Dropout(self.drop_rate)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())]))

        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

        # RF-Next
        self.prepare_rfsearch(pretrained_weights, rf_mode, kernel_cfgs, search_cfgs)
        if self.rf_mode not in ['rfsearch', 'rfmultiple']:
            for n, p, in self.named_parameters():
                if 'sample_weights' in n:
                    p.requires_grad = False
    
    def prepare_rfsearch(self, pretrained_weights, rf_mode, kernel_cfgs, search_cfgs):
        self.rf_mode = rf_mode
        self.pretrained_weights = pretrained_weights
        assert self.rf_mode in ['rfsearch', 'rfsingle', 'rfmultiple', 'rfmerge'], \
            "rf_mode should be in ['rfsearch', 'rfsingle', 'rfmultiple', 'rfmerge']."
        if pretrained_weights is None or not os.path.exists(pretrained_weights):
            checkpoint = None
        else:
            checkpoint = torch.load(pretrained_weights, map_location='cpu')
            checkpoint = checkpoint_filter_fn(checkpoint, self)
            # Remove the prefix in checkpint, e.g., 'backbone' and 'module', 
            # to guarantee the matching between 'checkpoint' and 'model.state_dict'.            
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            checkpoint = {k.replace('backbone.', ''): v for k, v in checkpoint.items()}
            for name in list(checkpoint.keys()):
                if name.endswith('counter') or name.endswith('current_search_step'):
                    # Do not load pretrained buffer of counter and current_step!!!!!!!
                    print(f"RF-Next: Removing key {name} from pretrained checkpoint")
                    del checkpoint[name]

            # Remove the parameters with mismatched shape from checkpoint         
            for name, module in self.named_parameters():
                if name in checkpoint and module.shape != checkpoint[name].shape:
                    print(f"RF-Next: Removing key {name} from pretrained checkpoint")
                    del checkpoint[name]
            # Load the pretrained weights for a rfconv. 
            # The pretarined weights are obtained after rfseach.
            msg = self.load_state_dict(checkpoint, strict=False)
            missing_keys = list(msg.missing_keys)
            missing_keys = list(filter(lambda x: not x.endswith('.counter') and not x.endswith('.current_search_step'), missing_keys))
            print('RF-Next: RF-Next init, missing keys: {}'.format(missing_keys))

        print('RF-Next: convert rfconv.')
        # Convert conv to rfconv
        def convert_rfconv(module, prefix):
            module_output = module
            if isinstance(module, RFConv2d):
                if kernel_cfgs is not None:
                    kernel = kernel_cfgs[prefix]
                else:
                    kernel = module.kernel_size
                if checkpoint is not None:
                    module_pretrained = dict()
                    # Load the pretrained weights for a rfconv. 
                    # The pretarined weights are obtained after rfseach.
                    for k in checkpoint.keys():
                        if k.startswith(prefix):
                            module_pretrained[k.replace('{}.'.format(prefix), '')] = checkpoint[k]
                else:
                    module_pretrained = None
                if isinstance(kernel, int):
                    kernel = (kernel, kernel)
                module_output = RFConv2d(
                    in_channels=module.in_channels, 
                    out_channels=module.out_channels, 
                    kernel_size=kernel,
                    stride=module.stride, 
                    padding=(
                        get_padding(kernel[0], module.stride[0], module.dilation[0]), 
                        get_padding(kernel[1], module.stride[1], module.dilation[1])),
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=hasattr(module, 'bias'),
                    rf_mode=self.rf_mode,
                    pretrained=module_pretrained,
                    **search_cfgs
                )
                
            for name, child in module.named_children():
                fullname = name
                if prefix != '':
                    fullname = prefix + '.' + name
                    # Replace the conv with rfconv。
                module_output.add_module(name, convert_rfconv(child, fullname))
            del module
            return module_output

        convert_rfconv(self, '')

        if self.rf_mode == 'rfmerge':
            # Show the kernel sizes after rfmerge。
            rfmerge = dict()
            for name, module in self.named_modules():
                if isinstance(module, RFConv2d):
                    rfmerge[name] = module.kernel_size
            
            print('Merged structure:')
            print(rfmerge)
        print('RF-Next: convert done.')

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm_pre', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        if global_pool is not None:
            self.head.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.head.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.head.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        # NOTE nn.Sequential in head broken down since can't call head[:-1](x) in torchscript :(
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)
        return x if pre_logits else self.head.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.zeros_(module.bias)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']
    out_dict = {}
    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        if ('current_search_step' in k) or ('counter' in k):
            continue
        out_dict[k] = v
    return out_dict


def _create_rfconvnext(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        RFConvNeXt, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs)
    return model


def rfconvnext_tiny(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    model = _create_rfconvnext('convnext_tiny', pretrained=pretrained, **model_args)
    return model


def rfconvnext_small(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    model = _create_rfconvnext('convnext_small', pretrained=pretrained, **model_args)
    return model


def rfconvnext_base(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_rfconvnext('convnext_base', pretrained=pretrained, **model_args)
    return model


def rfconvnext_large(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = _create_rfconvnext('convnext_large', pretrained=pretrained, **model_args)
    return model
