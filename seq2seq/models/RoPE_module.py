import torch
import torch.nn as nn
import numpy as np

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim_embed, max_seq_len):
        super(RotaryPositionalEncoding, self).__init__()
        self.dim_embed = dim_embed
        self.max_seq_len = max_seq_len

        position_enc = np.array(
            [[[pos / np.power(10000, 2 * (j // 2) / dim_embed) for j in range(dim_embed)] for pos in range(self.max_seq_len)]]
        )
        sinusoidal_pos = torch.empty(1, self.max_seq_len, dim_embed)
        sentinel = dim_embed // 2 if dim_embed % 2 == 0 else (dim_embed // 2) + 1
        sinusoidal_pos[:, :, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, :, 0::2]))
        sinusoidal_pos[:, :, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, :, 1::2]))

        # sin [batch_size, max_seq_len, dim_embed//2]
        # cos [batch_size, max_seq_len, dim_embed//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        self.sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        self.cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)

    def forward(self, x):
        # x: [batch_size, seq_len, dim_embed]
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_x = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(
            x
        )
        x = x * self.cos_pos[:, :x.size(1), :] + rotate_half_x[:, :x.size(1), :] * self.sin_pos[:, :x.size(1), :]
        return x