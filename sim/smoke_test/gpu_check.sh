#!/usr/bin/env bash
# usage: gpu_check.sh [nvidia]  — run headless sim 12 s, report GPU use + image rate
cd "$(dirname "$0")"; source /opt/ros/jazzy/setup.bash
[ "${1:-}" = nvidia ] && export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
setsid gz sim -s -r --headless-rendering rgbd_smoke.sdf > gz_check.log 2>&1 & P=$!
sleep 10
echo "nvidia procs:"; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; nvidia-smi | grep -i -E "ruby|gz" || echo "(no gz on nvidia)"
echo "rate:"; timeout 6 gz topic -t /rgbd/image -e 2>/dev/null | grep -c "^header" 
kill -- -$P; sleep 1
grep -i -E "egl|error" gz_check.log | head -5
