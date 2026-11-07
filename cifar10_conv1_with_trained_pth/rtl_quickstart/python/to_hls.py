"""hls4ml -> HLS C++ project. Requires Vitis HLS for actual RTL synthesis.
NOTE: uses fixed-point arithmetic; it is NOT an IEEE-754 FP32 equivalent.
"""
import argparse
from pathlib import Path
import torch
from small_model import SmallVGG6

p=argparse.ArgumentParser()
p.add_argument('--part',required=True,help='EXACT FPGA part e.g. xc7a35tcpg236-1, not just Artix-7')
p.add_argument('--checkpoint',default='small_cifar10.pth')
p.add_argument('--synth',action='store_true',help='run Vitis HLS synthesis to generate RTL')
p.add_argument('--precision',default='fixed<16,6>')
p.add_argument('--reuse',type=int,default=16)
a=p.parse_args()

import hls4ml
m=SmallVGG6()
m.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
m.eval()
# hls4ml supports PyTorch conv/pool/BN/linear but converter/version limitations may apply.
config=hls4ml.utils.config_from_pytorch_model(
    m, input_shape=(3,32,32), granularity='name',
    default_precision=a.precision, default_reuse_factor=a.reuse, backend='Vitis'
)
config['Model']['Strategy']='Resource'
project=hls4ml.converters.convert_from_pytorch_model(
    m, hls_config=config, output_dir='cnn_hls', project_name='cnn_hls',
    backend='Vitis', part=a.part, clock_period=10, io_type='io_stream'
)
project.write()
print('Generated HLS C++ project: cnn_hls/')
print('IMPORTANT: RTL Verilog files appear only AFTER Vitis HLS synthesis.')
if a.synth:
    print('Starting Vitis HLS synthesis; this requires Vitis HLS installed and on PATH')
    result = project.build(synth=True)
    print('Synthesis finished. Result:', result)
