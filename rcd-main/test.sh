#!/bin/bash
set -e

ENV_PATH="/opt/miniconda3/envs/qwen32b"
SITE_PKGS="$ENV_PATH/lib/python3.10/site-packages"

# 你之前 ln 的所有目标文件（必须精确匹配）
LINKED_FILES=(
    "$SITE_PKGS/pyAgrum/lib/image.py"
    "$SITE_PKGS/causallearn/search/ConstraintBased/FCI.py"
    "$SITE_PKGS/causallearn/utils/Fas.py"
    "$SITE_PKGS/causallearn/utils/PCUtils/SkeletonDiscovery.py"
    "$SITE_PKGS/causallearn/graph/GraphClass.py"
)

echo "=== 正在清理手动软链接 ==="

for file in "${LINKED_FILES[@]}"; do
    if [ -L "$file" ]; then
        echo "删除软链接: $file -> $(readlink -f "$file")"
        rm -f "$file"
    elif [ -f "$file" ]; then
        echo "跳过（非软链接）: $file"
    else
        echo "跳过（不存在）: $file"
    fi
done

# 可选：删除空目录（谨慎）
echo "=== 清理空目录（可选）==="
find "$SITE_PKGS/pyAgrum/lib" \
     "$SITE_PKGS/causallearn/search/ConstraintBased" \
     "$SITE_PKGS/causallearn/utils/PCUtils" \
     -type d -empty -delete 2>/dev/null || true

echo "=== 软链接清理完成！==="
echo "建议：重新安装原始包"
echo "pip install pyagrum causallearn"