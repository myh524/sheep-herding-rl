#!/usr/bin/env bash
# RTX 50 系（如 5060 Laptop）在专有内核模块下会出现：
#   NVRM: requires use of the NVIDIA open kernel modules
#   RmInitAdapter failed → nvidia-smi: No devices were found
# 本脚本切换到 nvidia-driver-580-open + 与当前内核匹配的 open 内核模块。

set -euo pipefail

if [[ "${EUID:-0}" -ne 0 ]]; then
  echo "请使用 root 运行: sudo bash $0"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
KVER="$(uname -r)"
MOD_OPEN="linux-modules-nvidia-580-open-${KVER}"
OBJ_OPEN="linux-objects-nvidia-580-open-${KVER}"

if ! apt-cache show "$MOD_OPEN" &>/dev/null; then
  echo "未找到与当前内核匹配的包: $MOD_OPEN"
  echo "请先: sudo apt update && sudo apt install linux-generic-hwe-22.04（或升级内核），再重新运行本脚本。"
  echo "或改为安装: sudo apt install -y nvidia-driver-580-open linux-modules-nvidia-580-open-generic-hwe-22.04"
  exit 1
fi

apt-get update
apt-get install -y nvidia-driver-580-open "$MOD_OPEN" "$OBJ_OPEN"

# 卸载专有内核模块包，避免与 open 模块并存时加载错误驱动
apt-get remove -y \
  "linux-modules-nvidia-580-${KVER}" \
  "linux-objects-nvidia-580-${KVER}" \
  linux-modules-nvidia-580-generic-hwe-22.04 \
  2>/dev/null || true

apt-get autoremove -y

echo ""
echo "安装完成。请重启: sudo reboot"
echo "重启后执行 nvidia-smi 应能看到 GPU。"
