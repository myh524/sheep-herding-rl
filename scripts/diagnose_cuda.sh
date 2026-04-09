#!/usr/bin/env bash
# 诊断「nvidia-smi 正常但 torch.cuda 不可用」：检查驱动类型、Prime、cuInit、PyTorch。
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "=== 已安装的 NVIDIA 驱动元包 ==="
dpkg -l | grep -E '^ii\s+nvidia-driver' || true
echo ""
echo "=== 当前加载的内核模块（应看到 nvidia 来自 open 或 非 open）==="
if [[ -r /proc/driver/nvidia/version ]]; then
  cat /proc/driver/nvidia/version
else
  echo "(无法读取 /proc/driver/nvidia/version)"
fi
echo ""
echo "=== PRIME 模式（on-demand 时笔记本上 CUDA 可能异常）==="
prime-select query 2>/dev/null || echo "无 prime-select"
echo ""
echo "=== nvidia-smi（NVML）==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 || true
echo ""
echo "=== libcuda cuInit（0=成功，999=未知错误，多为驱动/模块问题）==="
python3 << 'PY'
import ctypes
lib = ctypes.CDLL("libcuda.so.1")
lib.cuInit.argtypes = [ctypes.c_uint]
lib.cuInit.restype = ctypes.c_int
code = int(lib.cuInit(0))
print("cuInit(0) ->", code, "(0 CUDA_SUCCESS, 999 cudaErrorUnknown)")
PY

echo ""
echo "=== PyTorch（优先使用项目 .venv）==="
PYBIN="$ROOT/.venv/bin/python"
if [[ -x "$PYBIN" ]]; then
  "$PYBIN" -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())" 2>&1
else
  python3 -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())" 2>&1 || echo "未安装 torch 或无 .venv"
fi

echo ""
echo "若 cuInit=999 且使用 nvidia-driver-*-open：常见修复是改用闭源 metapackage nvidia-driver-580 并重启。"
