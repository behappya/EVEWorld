"""weightmap 变体 legacy_multilevel: 现行五档设计 (0.5/2/3/4/6), 即 t4g_weightmap 方案。

并列六选一, 无推荐; 详见 eveworld/pipeline/t4g_weightmap_variants.md。
"""

from .t4g_weightmap_variants import variant_config


config = variant_config('legacy_multilevel')
