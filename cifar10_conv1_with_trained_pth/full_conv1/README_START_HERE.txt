FULL CONV1 RTL QUICKSTART - original trained model included
====================================================
CHECKPOINT: ../vgg6_cifar10.pth (original 6-layer trained model)
LANGUAGE: Verilog-2005 with a SystemVerilog-compatible testbench invocation (-g2012)
NUMERICAL PRECISION: signed Q16.16 fixed point; THIS IS NOT FP32.
MODEL: Conv1 only, 64 filters x 3 input channels x 3x3, 32x32 output.
INPUT: deterministic normalized-like synthetic 3x32x32 test vector, not a CIFAR-10 photo.
PARAMETERS: Original conv1 weights and bias extracted from checkpoint.
OPERATION: convolution ONLY, including bias and zero padding. DOES NOT include BN1/ReLU.
TEST: golden outputs calculated from the same quantized inputs and weights by Python.

FAST SIMULATION (from full_conv1 directory):
  iverilog -g2012 -s tb_conv1_serial_q16 -o conv1_sim.vvp rtl/conv1_serial_q16.v tb/tb_conv1_serial_q16.v
  vvp conv1_sim.vvp
EXPECTED: CHECKED 1024 Conv1 outputs across 1 filters; errors=0; PASS

FULL FIRST CONVOLUTION (64 FILTERS; SLOWER):
  iverilog -g2012 -DALL_FILTERS -s tb_conv1_serial_q16 -o conv1_all.vvp rtl/conv1_serial_q16.v tb/tb_conv1_serial_q16.v
  vvp conv1_all.vvp
EXPECTED: CHECKED 65536 Conv1 outputs across 64 filters; errors=0; PASS

VIVADO SIMULATOR (from full_conv1 directory):
  xvlog rtl/conv1_serial_q16.v tb/tb_conv1_serial_q16.v
  xelab tb_conv1_serial_q16 -s conv1_sim
  xsim conv1_sim -runall

REGENERATE TEST VECTORS (only if updating input or checkpoint):
  python python/regenerate_mem.py ../vgg6_cifar10.pth
  (needs torch, numpy)

IMPORTANT: This simulation version uses initialized on-chip image and weight ROMs, generic combinational reads,
32-bit Q16.16 signed multiplication and 64-bit accumulations. Real device synthesis and timing closure
require adapting the ROMs to synchronous BRAM or streaming interfaces, introducing pipelines, constraints,
and verifying resource fit on the EXACT Artix-7 FPGA part. An FPGA with no external memory may not store
all 64 feature maps; outputs are streamed here rather than buffered. ChipWhisperer capture also needs
hardware-specific clock/reset/trigger and a suitable power sensing setup.

To use actual input CIFAR-10 images later, export their normalized CHW activations to the input .mem,
then regenerate the golden outputs.  To implement full Conv1+BN+ReLU, also export BN1 running stats and
learned affine parameters or fold BatchNorm into the convolution weights/biases first.

rtl_quickstart/ contains an earlier 27-tap MAC-only simulation example.
