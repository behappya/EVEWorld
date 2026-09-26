"""weightmap 变体 binary_3x: 六类区域统一 3x / 背景 1x 的二值设计。

并列六选一, 无推荐; 详见 eveworld/pipeline/t4g_weightmap_variants.md。
"""

from .t4g_weightmap_variants import variant_config


config = variant_config('binary_3x')
