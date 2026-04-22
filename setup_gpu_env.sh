#!/usr/bin/env bash
# setup_gpu_env.sh
# Appends GPU runtime library paths to ~/.bashrc.
# Run this once per machine (or per shell profile) before using HHLO with GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Build the library path
# ---------------------------------------------------------------------------

PATHS=()

# 1. HHLO project-local symlink dir (for nvshmem workaround)
SYMLINK_DIR="$SCRIPT_DIR/deps/pjrt/lib_symlinks"
if [[ -d "$SYMLINK_DIR" ]]; then
    PATHS+=("$SYMLINK_DIR")
fi

# 2. NVIDIA Python wheels (conda / pip / uv common locations)
search_nvidia_libs() {
    local base="$1"
    if [[ -d "$base" ]]; then
        for pkg in cu13 cudnn nccl nvshmem; do
            local libdir="$base/nvidia/$pkg/lib"
            if [[ -d "$libdir" ]]; then
                PATHS+=("$libdir")
            fi
        done
    fi
}

search_nvidia_libs "$HOME/miniconda3/lib/python3.13/site-packages"
search_nvidia_libs "$HOME/miniconda3/lib/python3.12/site-packages"
search_nvidia_libs "$HOME/miniconda3/lib/python3.11/site-packages"
search_nvidia_libs "$HOME/.conda/lib/python3.13/site-packages"
search_nvidia_libs "$HOME/.conda/lib/python3.12/site-packages"
search_nvidia_libs "$HOME/.conda/lib/python3.11/site-packages"
search_nvidia_libs "$HOME/.local/lib/python3.13/site-packages"
search_nvidia_libs "$HOME/.local/lib/python3.12/site-packages"
search_nvidia_libs "$HOME/.local/lib/python3.11/site-packages"

# 3. System CUDA toolkit (if present)
if [[ -d "/usr/local/cuda/targets/x86_64-linux/lib" ]]; then
    PATHS+=("/usr/local/cuda/targets/x86_64-linux/lib")
fi
if [[ -d "/usr/local/cuda/lib64" ]]; then
    PATHS+=("/usr/local/cuda/lib64")
fi

# ---------------------------------------------------------------------------
# Deduplicate while preserving order
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
# Append to ~/.bashrc (idempotent)
# ---------------------------------------------------------------------------

BASHRC="$HOME/.bashrc"
MARKER="# >>> HHLO GPU runtime library path"

if grep -qF "$MARKER" "$BASHRC" 2>/dev/null; then
    echo "HHLO GPU env block already found in $BASHRC. Skipping."
    echo "If you want to refresh it, remove the block between '$MARKER' and '# <<< HHLO GPU' in $BASHRC and rerun."
    exit 0
fi

cat >> "$BASHRC" << EOF

$MARKER
# Added by HHLO setup_gpu_env.sh on $(date -Iseconds)
# This makes the PJRT CUDA plugin and its dependencies discoverable.
export LD_LIBRARY_PATH="$NEW_PATH:\$LD_LIBRARY_PATH"
# <<< HHLO GPU
EOF

echo "Appended GPU library paths to $BASHRC"
echo ""
echo "Directories added:"
for p in "${UNIQ_PATHS[@]}"; do
    echo "  - $p"
done
echo ""
echo "Run 'source ~/.bashrc' or open a new terminal to apply the changes."
