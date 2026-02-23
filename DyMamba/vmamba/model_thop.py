import torch 
import torch.nn as nn
from thop import profile, clever_format

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vss_args = dict(
    in_chans=3, 
    patch_size=4, 
    depths=[2,2,4,2], 
    dims=64, 
    drop_path_rate=0.2)
decoder_args = dict(
    num_classes=3,
    deep_supervision=False, 
    features_per_stage=[64, 128, 256, 512],      
    drop_path_rate=0.2,
    d_state=16)

from SwinUMamba_DS import SwinUMambaD
model = SwinUMambaD(vss_args, decoder_args).to(device)

input_tensor = torch.randn(1, 3, 512, 512).to(device)
flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Parameters: {params / 1e6:.2f} M")