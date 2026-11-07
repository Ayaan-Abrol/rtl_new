RTL QUICKSTART FROM YOUR ACTUAL CIFAR-10 CHECKPOINT
==================================================

THE TWO MILESTONES ARE DIFFERENT:
A) Immediate: simulate a real Verilog convolution dot-product with 27 trained
   Conv1 weights from your existing 4.6 MB FP32 checkpoint. This starter uses
   Q16.16 FIXED-POINT, not IEEE-754 FP32, and is not a full CNN.
B) Later: train a NEW smaller CNN, use hls4ml to create C++ HLS source,
   then run Vitis HLS synthesis to produce FULL NETWORK RTL (if the CNN
   converts and the synthesis fits your Artix-7 device). This HLS flow also
   uses FIXED-POINT by default. Do not label either as FP32 RTL.

EXTRACT ZIP INTO YOUR EXISTING cifar-10-python DIRECTORY so you get:
  C:\Users\ayaan\Downloads\cifar-10-python\rtl_quickstart\...

OPEN VS CODE POWERSHELL TERMINAL:
  cd C:\Users\ayaan\Downloads\cifar-10-python\rtl_quickstart
  py python\generate_vectors.py ..\vgg6_cifar10.pth

SIMULATE WITH VIVADO XSIM (Vivado tools must be on PATH):
  xvlog rtl\conv1_mac.v tb\tb_conv1_mac.v
  xelab tb_conv1_mac -s conv1_sim
  xsim conv1_sim -runall

OR WITH ICARUS VERILOG (if installed):
  iverilog -g2012 -o conv1_tb.vvp rtl\conv1_mac.v tb\tb_conv1_mac.v
  vvp conv1_tb.vvp

EXPECTED:
  RESULT Q16.16 integer = 40836
  RESULT approximate = 0.623108
  PASS: RTL convolution dot-product matches fixed-point reference

THIS STARTER DOES NOT EXECUTE THE COMPLETE CNN. For a small CNN:
  py python\small_model.py
  py python\train_small.py --data .. --epochs 60
  # Writes small_cifar10.pth to the working directory.
  # It trains from scratch: cannot reuse 4.6MB checkpoint directly.

HLS CONVERSION (USE PYTHON 3.11 VENV, and have AMD Vitis HLS installed):
  py -3.11 -m venv .venv-hls
  .\.venv-hls\Scripts\python.exe -m pip install --upgrade pip
  .\.venv-hls\Scripts\python.exe -m pip install torch hls4ml numpy
  .\.venv-hls\Scripts\python.exe python\to_hls.py --part YOUR_EXACT_FPGA_PART

Important: 'Artix-7' by itself is NOT an exact FPGA part number.
If model conversion passes, this generates cnn_hls/ C++ HLS source.
To run synthesis and produce Verilog RTL, use the AMD Vitis HLS command
prompt or a supported configured command shell and run:
  .\.venv-hls\Scripts\python.exe python\to_hls.py --part YOUR_EXACT_FPGA_PART --synth

The '--synth' option will run Vitis HLS from Python after writing the project.
Depending on your version, hls4ml may require adjustments for unsupported
layers and target resources. Inspect the generated cnn_hls/ directories
for Verilog files after synthesis.

Read the accompanying response for hardware verification and FPGA implementation.
