#!/usr/bin/env bash
# IGR 权重图六变体总入口 (六变体并列, 无推荐): 一次体检 / 一次预计算全部六个变体。
# 与六个 t4g_weightmap_<variant>_launch.sh 配套; 本脚本只做体检与缓存预计算, 不提交训练。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAGI="/data/datasets/gagi"
WMAP_ROOT="${T4G_WMAP_ROOT:-${GAGI}/eve_v2_outputs/track4gen_probe}"
PYTHON="${PYTHON:-/home/jovyan/miniconda/envs/EVEWorld/bin/python}"
VARIANTS=(legacy_multilevel binary_3x path_loc_2x path_loc_3x interaction_2x interaction_3x)

run_check() {
  local v
  for v in "${VARIANTS[@]}"; do
    echo "=== check ${v} ==="
    bash "${HERE}/t4g_weightmap_${v}_launch.sh" check
  done
}

run_precompute() {
  T4G_WMAP_ROOT="${WMAP_ROOT}" \
    "${PYTHON}" "${HERE}/t4g_weightmap_variants_precompute.py" --variant all
}

run_list() {
  "${PYTHON}" "${HERE}/t4g_weightmap_variants.py" --list-variants
}

case "${1:-}" in
  check) run_check ;;
  precompute) run_precompute ;;
  list) run_list ;;
  *)
    cat <<EOF
IGR 权重图六变体 (并列, 无推荐)
  variants: ${VARIANTS[*]}
  cache:    ${WMAP_ROOT}/weightmap_cache_<variant>   (可用 T4G_WMAP_ROOT 覆盖)
  usage:    python eveworld/pipeline/t4g_weightmap_variants.py --list-variants

Commands:
  bash $(basename "$0") list         # 列出六变体的级别/区域/缓存目录
  bash $(basename "$0") check        # 依次跑六个 launcher 的 check (静态检查 + 自检 + 缓存体检)
  bash $(basename "$0") precompute   # 六个变体各写一份权重图缓存
  # 训练仍逐个变体自选提交 (六者并列, 无推荐):
  #   bash t4g_weightmap_legacy_multilevel_launch.sh submit
  #   bash t4g_weightmap_binary_3x_launch.sh submit
  #   bash t4g_weightmap_path_loc_2x_launch.sh submit
  #   bash t4g_weightmap_path_loc_3x_launch.sh submit
  #   bash t4g_weightmap_interaction_2x_launch.sh submit
  #   bash t4g_weightmap_interaction_3x_launch.sh submit
EOF
    ;;
esac
