"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.layers import *
from utils.nets.transdfnet import TransformerBlock
from functools import partial


DEFAULT_IN_CONV_KWARGS = {
                            'kernel_size': 3, 
                            'stride': 3,
                            'padding': '0',
                          }
DEFAULT_OUT_CONV_KWARGS = {
                            'kernel_size': 30, 
                            'stride': 30,
                            'padding': '0',
                           }
DEFAULT_MIXER_KWARGS = {
                            "type": "mhsa",
                            "head_dim": 16,
                            "use_conv_proj": True,
                            "kernel_size": 3,
                            "stride": 2,
                            "feedforward_style": "mlp",
                            "feedforward_ratio": 4,
                            "feedforward_drop": 0.0
                            }

class EspressoNet(nn.Module):
    """
    """
    def __init__(self, input_channels, 
                       input_size = 1200, 
                       depth = 0,
                       hidden_dim = 128,
                       feature_dim = 64,
                       special_toks = 0,
                       head_ratio = 4,
                       mixer_kwargs = DEFAULT_MIXER_KWARGS,
                       input_conv_kwargs = DEFAULT_IN_CONV_KWARGS,
                       output_conv_kwargs = DEFAULT_OUT_CONV_KWARGS,
                    **kwargs):
        super(EspressoNet, self).__init__()

        self.dummy_param = nn.Parameter(torch.empty(0))

        # # # #
        self.input_channels = input_channels
        self.input_conv_kwargs = input_conv_kwargs
        self.output_conv_kwargs = output_conv_kwargs

        self.input_size = input_size

        # (global) mixing op for transformer blocks
        if mixer_kwargs['type'] == 'mhsa':
            self.mixer = partial(MHSAttention, **mixer_kwargs)

        # (local) mixing op uses convolutions only
        elif "conv" in mixer_kwargs['type']:
            self.mixer = partial(ConvMixer, **mixer_kwargs)

        # No mixing op (signal processing is performed by MLPs only)
        else:
            self.mixer = partial(nn.Identity, **mixer_kwargs)


        self.depth = depth
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.special_toks = special_toks
        self.head_ratio = head_ratio

        if self.special_toks > 0:
            toks = nn.init.xavier_uniform_(torch.empty(self.hidden_dim, self.special_toks))
            self.extra_tokens = nn.Parameter(toks)
            self.special_fc = nn.Linear(self.hidden_dim, self.feature_dim)

        # construct the model using the selected params
        self.__build_model()
        # # # #

    def __build_model(self):
        """Construct the model layers
        """
        # define first conv. layer that handles feature embedding
        conv_embed = nn.Conv1d(self.input_channels, self.hidden_dim, 
                               **self.input_conv_kwargs)
        block_list = [conv_embed]

        # add transformer layers if they are enabled for the stage
        block_func = partial(TransformerBlock, 
                                    dim = self.hidden_dim, 
                                    token_mixer = self.mixer,
                                    skip_toks=self.special_toks+1,
                             )
        block_list += [block_func() for _ in range(self.depth)]

        self.blocks = nn.ModuleList(block_list)

        self.windowing = nn.Conv1d(self.hidden_dim, self.hidden_dim * self.head_ratio,
                                   groups = self.hidden_dim,
                                   **self.output_conv_kwargs)

        self.pred = nn.Sequential(
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim * self.head_ratio),
                nn.Linear(self.hidden_dim * self.head_ratio, self.feature_dim),
                )
        
        self.embed = nn.Embedding(4, self.hidden_dim)

    def get_device(self):
        return self.dummy_param.get_device()

    def forward(self, x, 
                x_proto = None,
            sample_sizes = None,
            return_toks = False,
            *args, **kwargs):
        """forward input features through the model
        """
        # add channel dim if necessary
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        # clip sample length to maximum supported size and pad with zeros if necessary
        size_dif = x.shape[-1] - self.input_size
        if x.shape[-1] > self.input_size:
            x = x[..., :self.input_size]
        elif size_dif < 0:
            x = F.pad(x, (0,abs(size_dif)))

        # padding can be ignored during self-attention if configured with pad_masks
        # note: padding-aware self-attention does not seem to be improve performance, but reduces computation efficiency
        pad_masks = None
        if sample_sizes is not None:
            pad_masks = torch.stack([torch.cat((torch.zeros(min(s, self.input_size)), 
                                                torch.ones(max(self.input_size-s, 0)))) for s in sample_sizes])
            pad_masks = pad_masks.to(x.get_device())
            pad_masks = pad_masks.unsqueeze(1)

        # apply conv. and transformer layers
        x = self.blocks[0](x)
        
        if x_proto is not None:
            x_proto = self.embed(x_proto).unsqueeze(-1)
            x = torch.cat((x_proto, x), dim=2)

        if self.special_toks > 0:
            special_toks_in = self.extra_tokens.unsqueeze(0).expand(x.shape[0],-1,-1)
            x = torch.cat((special_toks_in, x), dim=2)
            
        for i,block in enumerate(self.blocks[1:]):
            x = block(x)

        if self.special_toks > 0:
            special_toks_out = x[:,:,:self.special_toks].permute(0,2,1)
            special_toks_out = self.special_fc(special_toks_out).permute(0,2,1)

            x = x[:,:,self.special_toks:]
            if x_proto is not None:
                x = x[:,:,1:]

        # apply windowing feature prediction
        x = self.windowing(x).permute(0,2,1)
        pred_windows = self.pred(x).permute(0,2,1)

        if self.special_toks > 0 and return_toks:
            return pred_windows, special_toks_out.flatten(start_dim=1)
        else:
            return pred_windows


