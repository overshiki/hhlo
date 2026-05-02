#!/usr/bin/env bash
# setup_gpu_env.sh
# Appends GPU runtime library paths to ~/.bashrc.
# Run this once per machine (or per shell profile) before using HHLO with GPU.
#
# Usage:
#   ./setup_gpu_env.sh          # idempotent: skips if block already exists
#   ./setup_gpu_env.sh --force  # regenerate the block even if it exists

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORCE=false

for arg in "$@"; do
    if [[ "$arg" == "--force" ]]; then
        FORCE=true
    fi
done

# ---------------------------------------------------------------------------
# 1. Detect the active Python interpreter
# ---------------------------------------------------------------------------
# We collect candidates and prefer the one that already has the NVIDIA
# wheels installed.  This avoids accidentally using system python3 when
# conda base (or a virtualenv) already has the packages.

find_python() {
    local candidates=()

    if command -v python &> /dev/null; then
        candidates+=("$(command -v python)")
    fi
    if [[ -x "$HOME/miniconda3/bin/python" ]]; then
        candidates+=("$HOME/miniconda3/bin/python")
    fi
    if command -v python3 &> /dev/null; then
        candidates+=("$(command -v python3)")
    fi
    if [[ -x "$HOME/miniconda3/bin/python3" ]]; then
        candidates+=("$HOME/miniconda3/bin/python3")
    fi

    # Prefer a candidate that already has the required NVIDIA packages
    for cand in "${candidates[@]}"; do
        local sp
        sp=$("$cand" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
        if [[ -n "$sp" && -d "$sp/nvidia/cudnn/lib" && -d "$sp/nvidia/nccl/lib" && -d "$sp/nvidia/nvshmem/lib" ]]; then
            echo "$cand"
            return
        fi
    done

    # No candidate has the packages yet — return the first available one
    if [[ ${#candidates[@]} -gt 0 ]]; then
        echo "${candidates[0]}"
    fi
}

PYTHON=$(find_python || true)

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: No Python interpreter found (tried: python, python3, ~/miniconda3/bin/python)."
    exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

# ---------------------------------------------------------------------------
# 2. Discover the Python site-packages directory
# ---------------------------------------------------------------------------

SITE_PACKAGES=$("$PYTHON" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)

if [[ -z "$SITE_PACKAGES" ]]; then
    echo "ERROR: Could not determine site-packages directory for $PYTHON."
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Auto-install missing NVIDIA wheels
# ---------------------------------------------------------------------------

PACKAGES_TO_INSTALL=()

if [[ ! -d "$SITE_PACKAGES/nvidia/cudnn/lib" ]]; then
    PACKAGES_TO_INSTALL+=("nvidia-cudnn-cu13")
fi
if [[ ! -d "$SITE_PACKAGES/nvidia/nccl/lib" ]]; then
    PACKAGES_TO_INSTALL+=("nvidia-nccl-cu13")
fi
if [[ ! -d "$SITE_PACKAGES/nvidia/nvshmem/lib" ]]; then
    PACKAGES_TO_INSTALL+=("nvidia-nvshmem-cu13")
fi

if [[ ${#PACKAGES_TO_INSTALL[@]} -gt 0 ]]; then
    echo ""
    echo "Missing NVIDIA libraries detected. Installing: ${PACKAGES_TO_INSTALL[*]}"
    "$PYTHON" -m pip install "${PACKAGES_TO_INSTALL[@]}"
    echo "Installation complete."
else
    echo "All required NVIDIA Python wheels are already installed."
fi

# Re-discover site-packages in case pip installed to a slightly different path
# (e.g. user site vs system site on some distros).
SITE_PACKAGES=$("$PYTHON" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [[ -z "$SITE_PACKAGES" ]]; then
    echo "ERROR: Could not re-determine site-packages after installation."
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Build the library path
# ---------------------------------------------------------------------------

PATHS=()

# 4a. HHLO project-local symlink dir (for nvshmem workaround)
SYMLINK_DIR="$SCRIPT_DIR/deps/pjrt/lib_symlinks"
mkdir -p "$SYMLINK_DIR"
PATHS+=("$SYMLINK_DIR")

# 4b. NVIDIA Python wheels (dynamically discovered from active Python)
if [[ -d "$SITE_PACKAGES" ]]; then
    for pkg in cu13 cudnn nccl nvshmem; do
        libdir="$SITE_PACKAGES/nvidia/$pkg/lib"
        if [[ -d "$libdir" ]]; then
            PATHS+=("$libdir")
        fi
    done
fi

# 4c. System CUDA toolkit
if [[ -d "/usr/local/cuda/targets/x86_64-linux/lib" ]]; then
    PATHS+=("/usr/local/cuda/targets/x86_64-linux/lib")
fi
if [[ -d "/usr/local/cuda/lib64" ]]; then
    PATHS+=("/usr/local/cuda/lib64")
fi

# 4d. CUPTI (often located in extras/CUPTI, not the main lib64 dir)
for cupti_dir in /usr/local/cuda/extras/CUPTI/lib64 /usr/local/cuda-*/extras/CUPTI/lib64; do
    if [[ -d "$cupti_dir" ]]; then
        PATHS+=("$cupti_dir")
    fi
done

# ---------------------------------------------------------------------------
# 5. NVSHMEM compatibility symlink workaround
# ---------------------------------------------------------------------------
# The PJRT CUDA plugin (as of nightly-2026-04-20) links against
# nvshmem_transport_ibrc.so.4, but the current nvidia-nvshmem-cu13 wheel
# ships .so.5.  Create a project-local compatibility symlink so ldd resolves.

NVSHMEM_LIB="$SITE_PACKAGES/nvidia/nvshmem/lib"
if [[ -d "$NVSHMEM_LIB" ]]; then
    if [[ ! -e "$NVSHMEM_LIB/nvshmem_transport_ibrc.so.4" && -e "$NVSHMEM_LIB/nvshmem_transport_ibrc.so.5" ]]; then
        echo ""
        echo "Creating NVSHMEM compatibility symlink: nvshmem_transport_ibrc.so.4 -> .so.5"
        ln -sf "nvshmem_transport_ibrc.so.5" "$NVSHMEM_LIB/nvshmem_transport_ibrc.so.4"
    fi
    # Also symlink into the project-local dir as a fallback
    if [[ ! -e "$SYMLINK_DIR/nvshmem_transport_ibrc.so.4" ]]; then
        ln -sf "$NVSHMEM_LIB/nvshmem_transport_ibrc.so.5" "$SYMLINK_DIR/nvshmem_transport_ibrc.so.4" || true
    fi
fi

# ---------------------------------------------------------------------------
# 6. Deduplicate while preserving order
# ---------------------------------------------------------------------------

declare -A SEEN
UNIQ_PATHS=()
for p in "${PATHS[@]}"; do
    if [[ -z "${SEEN[$p]:-}" ]]; then
        SEEN[$p]=1
        UNIQ_PATHS+=("$p")
    fi
done

if [[ ${#UNIQ_PATHS[@]} -eq 0 ]]; then
    echo "No GPU library directories found. Nothing to append."
    exit 1
fi

# Build the colon-separated path string
NEW_PATH=""
for p in "${UNIQ_PATHS[@]}"; do
    if [[ -z "$NEW_PATH" ]]; then
        NEW_PATH="$p"
    else
        NEW_PATH="$NEW_PATH:$p"
    fi
done

# ---------------------------------------------------------------------------
# 7. Append to ~/.bashrc (idempotent by default, overridable with --force)
# ---------------------------------------------------------------------------

BASHRC="$HOME/.bashrc"
MARKER="# >>> HHLO GPU runtime library path"

if grep -qF "$MARKER" "$BASHRC" 2>/dev/null && [[ "$FORCE" == false ]]; then
    echo ""
    echo "HHLO GPU env block already found in $BASHRC. Skipping."
    echo "If you want to refresh it (e.g. after installing new packages), rerun with --force."
    exit 0
fi

# Remove old block if it exists so we can regenerate it
if grep -qF "$MARKER" "$BASHRC" 2>/dev/null; then
    tmpfile=$(mktemp)
    awk '/# >>> HHLO GPU runtime library path/{skip=1} !skip{print} /# <<< HHLO GPU/{skip=0}' "$BASHRC" > "$tmpfile"
    mv "$tmpfile" "$BASHRC"
fi

cat >> "$BASHRC" << EOF

$MARKER
# Added by HHLO setup_gpu_env.sh on $(date -Iseconds)
# This makes the PJRT CUDA plugin and its dependencies discoverable.
export LD_LIBRARY_PATH="$NEW_PATH:\$LD_LIBRARY_PATH"
# <<< HHLO GPU
EOF

echo ""
echo "Appended GPU library paths to $BASHRC"
echo ""
echo "Directories added:"
for p in "${UNIQ_PATHS[@]}"; do
    echo "  - $p"
done
echo ""
echo "Run 'source ~/.bashrc' or open a new terminal to apply the changes."

# ---------------------------------------------------------------------------
# 8. Validate PJRT CUDA plugin dependencies
# ---------------------------------------------------------------------------

PJRT_CUDA="$SCRIPT_DIR/deps/pjrt/libpjrt_cuda.so"
if [[ -f "$PJRT_CUDA" ]]; then
    echo ""
    echo "Validating PJRT CUDA plugin dependencies..."
    MISSING=$(LD_LIBRARY_PATH="$NEW_PATH:${LD_LIBRARY_PATH:-}" ldd "$PJRT_CUDA" 2>/dev/null | grep "not found" || true)
    if [[ -n "$MISSING" ]]; then
        echo "WARNING: Some dependencies are still missing:"
        echo "$MISSING"
        echo ""
        echo "If you see nvshmem_transport_ibrc.so.4 above, the symlink workaround failed."
        echo "You may need to install a different PJRT CUDA plugin version."
        exit 1
    else
        echo "All PJRT CUDA plugin dependencies resolved."
    fi
fi
