"""Re-generate first-filter ROM and deterministic test patch from original .pth.
Run from rtl_quickstart folder:
  py python/generate_vectors.py path/to/vgg6_cifar10.pth
"""
from pathlib import Path
import sys
import numpy as np
import torch

root = Path(__file__).resolve().parent.parent
checkpoint = Path(sys.argv[1]) if len(sys.argv)>1 else root/'vgg6_cifar10.pth'
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
weights = state['conv1.weight'][0].detach().numpy().reshape(-1)
assert weights.size==27
# Synthetic normalized RGB 3x3 patch; channel-major order, same as PyTorch Conv2d.
rng = np.random.default_rng(123)
uint8_patch = rng.integers(0, 256, size=(3,3,3), dtype=np.uint8)
mean = np.array([0.4914,0.4822,0.4465], dtype=np.float64)[:,None,None]
std  = np.array([0.2470,0.2435,0.2616], dtype=np.float64)[:,None,None]
patch = ((uint8_patch.astype(np.float64)/255 - mean)/std).reshape(-1)
SCALE = 65536
wq = np.rint(weights.astype(np.float64)*SCALE).astype(np.int64)
xq = np.rint(patch*SCALE).astype(np.int64)
# hardware calculates sum of products with no intermediate rounding, then right shifts 16
acc = int(np.sum(wq*xq))
expected = acc >> 16
assert -2**31 <= expected < 2**31

for out_name, values in [('conv1_filter0_q16.mem',wq),('test_patch_q16.mem',xq)]:
    with (root/'mem'/out_name).open('w') as f:
        for v in values:
            f.write(f'{int(v)&0xffffffff:08X}\n')

p = root/'tb'/'tb_conv1_mac.v'
txt = p.read_text()
# Update single reference constant in the testbench for new weights.
import re
txt = re.sub(r'localparam signed \[31:0\] EXPECTED_Q16 = 32\'sd-?\d+;',
             f"localparam signed [31:0] EXPECTED_Q16 = 32'sd{expected};", txt)
p.write_text(txt)
print('Wrote 27 weights, 27 patch pixels, and expected result')
print('Expected fixed-point integer:',expected)
print('Expected approximate float:',expected/SCALE)
print('FP32 software dot product:', float(np.dot(weights.astype(np.float64),patch)))
