#!/bin/bash
set -euo pipefail

cd /home/eterrescaballe/bta_paper

if [[ "${CONDA_DEFAULT_ENV:-}" != "sm" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate sm
fi

install_root="${1:-$PWD/.local/mujoco}"
archive="${TMPDIR:-/tmp}/mujoco210-linux-x86_64.tar.gz"
url="https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
py_src="$PWD/.local/mujoco_py"
compat_root="$PWD/.local/mujoco_compat"
gcrypt_rpm="${TMPDIR:-/tmp}/libgcrypt11-1.4.0-15.el6.x86_64.rpm"
gcrypt_url="https://rpmfind.net/linux/atrpms/sl6-x86_64/atrpms/testing/libgcrypt11-1.4.0-15.el6.x86_64.rpm"

mkdir -p "$install_root"
if [[ -d "$install_root/mujoco210" ]]; then
  echo "Using existing MuJoCo install at $install_root/mujoco210"
else
  if [[ -s "$archive" ]]; then
    echo "Using existing MuJoCo archive at $archive"
  else
    echo "Downloading MuJoCo 2.1 to $archive"
    wget -O "$archive" "$url"
  fi
  echo "Extracting to $install_root"
  tar -xzf "$archive" -C "$install_root"
fi

echo "Copying mujoco_py into writable project-local cache"
python - <<'PY'
import importlib.util
import pathlib
import shutil

spec = importlib.util.find_spec("mujoco_py")
if spec is None or spec.origin is None:
    raise SystemExit("mujoco_py is not installed in the active Python environment")

src = pathlib.Path(spec.origin).parent
dst_root = pathlib.Path(".local/mujoco_py")
dst = dst_root / "mujoco_py"
dst_root.mkdir(parents=True, exist_ok=True)
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
(dst / "generated").mkdir(exist_ok=True)
print(f"Copied {src} -> {dst}")
PY

if [[ -e "$compat_root/usr/lib64/libgcrypt.so.11" ]]; then
  echo "Using existing libgcrypt.so.11 at $compat_root/usr/lib64"
else
  mkdir -p "$compat_root"
  if [[ -s "$gcrypt_rpm" ]]; then
    echo "Using existing libgcrypt11 RPM at $gcrypt_rpm"
  else
    echo "Downloading libgcrypt11 compatibility RPM to $gcrypt_rpm"
    wget -O "$gcrypt_rpm" "$gcrypt_url"
  fi
  echo "Extracting libgcrypt11 to $compat_root"
  (cd "$compat_root" && rpm2cpio "$gcrypt_rpm" | cpio -id)
fi

echo "MuJoCo installed at $install_root/mujoco210"
echo "Use:"
echo "  export PYTHONPATH=$py_src:\${PYTHONPATH:-}"
echo "  export MUJOCO_PY_MUJOCO_PATH=$install_root/mujoco210"
echo "  export CPATH=/usr/include:\${CONDA_PREFIX}/include:\${CPATH:-}"
echo "  export C_INCLUDE_PATH=/usr/include:\${CONDA_PREFIX}/include:\${C_INCLUDE_PATH:-}"
echo "  export CC=/usr/bin/gcc"
echo "  export CXX=/usr/bin/g++"
echo "  export LIBRARY_PATH=/usr/lib64:\${CONDA_PREFIX}/lib:\${LIBRARY_PATH:-}"
echo "  export LDFLAGS=\"-L/usr/lib64 -L\${CONDA_PREFIX}/lib \${LDFLAGS:-}\""
echo "  export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-}:$install_root/mujoco210/bin:$compat_root/usr/lib64:\${CONDA_PREFIX}/lib:/usr/lib64"
echo "  export MUJOCO_PY_FORCE_CPU=\${MUJOCO_PY_FORCE_CPU:-1}"
