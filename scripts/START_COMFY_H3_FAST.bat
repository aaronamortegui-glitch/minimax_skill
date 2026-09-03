@echo off
REM ============================================================
REM  ComfyUI with the measured speed optimizations for MiniMax H3
REM  RTX 5090 Laptop (sm_120) - torch 2.12+cu130 - 64 GB RAM
REM ============================================================
REM  --use-sage-attention   SageAttention 2.2. The single biggest win:
REM                         quantized attention kernels.
REM                         Measured: 1.62x faster, no quality loss.
REM  --fast fp16_accumulation cublas_ops
REM                         fp16 accumulation + cuBLAS kernels.
REM                         Do NOT add fp8_matrix_mult: the weights are
REM                         already int8_convrot and it degrades them.
REM  --cache-ram 40         caches nodes in RAM so the 15.7 GB text
REM                         encoder is not reloaded between runs.
REM  --vram-headroom 2      deliberately keeps 2 GB of VRAM free. Without
REM                         it, long video references push VRAM to 98%,
REM                         the allocator runs out of slack, and ComfyUI
REM                         sticks swapping weights forever. It does not
REM                         error out, it just hangs.
REM  NOTE: --high-ram is incompatible with --cache-ram, pick one.
REM ============================================================

.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build ^
  --use-sage-attention ^
  --fast fp16_accumulation cublas_ops ^
  --cache-ram 40 ^
  --vram-headroom 2

echo.
echo If ComfyUI did not start, try removing --use-sage-attention first
echo and then --fast, to isolate which one is the problem.
pause
