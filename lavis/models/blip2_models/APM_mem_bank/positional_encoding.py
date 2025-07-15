# generate a positional encoding for a heuight, width , and a particular time slice 

import os 
import numpy as np 
import torch 
from einops import rearrange, reduce, repeat 
import glob 
import glob 
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
# import timm
from transformers import AutoImageProcessor, AutoModel
import torch

import numpy as np
import torch
import torch.nn as nn


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.dtype_override = dtype_override
        self.channels = channels
    
    
    # generates positional encoding from start_time to start_time + T
    # should be helpful in a streaming setting
    def forward(self, tensor, start_time = 0):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        # pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        #x here is the number of frames for which positional encoding is genearted 
        
        pos_x = torch.arange(start_time, start_time + x, device=tensor.device, dtype=self.inv_freq.dtype)
        #print("pos_x", pos_x)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        # print("aa", pos_x.device, self.inv_freq.device)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        # print("sin inp_x", sin_inp_x)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc



if __name__ == "__main__":
    
    T,H,W = 32, 32 ,32
    t = 3  # so generate a positional encoding from t to t+T01
    d = 1024
    pos_3d = PositionalEncoding3D(d)
    fw = torch.zeros(1,T,H,W,d)
    pos_3d = pos_3d(fw, start_time = t).squeeze(0)
    pos_3d = rearrange(pos_3d, 't h w c -> (t h w) c')
    pos_3d = pos_3d.cuda() # h*w*l , 1024
    
    print("pos_3d.shape",pos_3d.shape)