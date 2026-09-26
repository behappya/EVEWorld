# IGR 空间权重图六变体 (`t4g_weightmap_*`)

本文件说明论文 App. 表 `tab:weightmap_design`（`paper/appme1.tex:1092-1212`）里比较过的
**六种 IGR 空间权重设计**在本仓库中的可运行实现：一个变体注册表 + 一个预计算 CLI
+ 六份训练 config（薄壳）+ 六个提交脚本 + 一个总入口。

**六者是并列的设计选项，本仓库不给出任何"推荐 / 默认 / 更好"的倾向**，选用哪一种由使用者决定。
唯一的硬约束是 `legacy_multilevel` 必须与仓库原有实现（`t4g_weightmap.build_weightmap`）
**逐位一致**，因为它是现行 `t4g_weightmap.py` / `t4g_weightmap_precompute.py` 的行为契约。

---

## 1. 论文表 -> 本仓库变体

对照 `paper/appme1.tex:1114-1211` 的表体（Coverage/MLR/IF 三列数字照抄自该表）：

| 论文 Design 行 | 变体名 | Path/Loc. | Arm/Grip. | Paste | Base | Emph. | Coverage (中位) | MLR ↓ | IF ↑ |
|---|---|---|---|---|---|---|---|---|---|
| Legacy multi-level | `legacy_multilevel` | ✓ | ✓ | ✓ | 0.5 | 2/3/4/6 | 0.2679 | 2.18 | 60.31 |
| Binary-3 | `binary_3x` | ✓ | ✓ | ✓ | 1 | 3 | 0.2685 | 2.21 | 60.26 |
| Path+Loc-2x | `path_loc_2x` | ✓ | | ✓ | 1 | 2 | 0.2380 | 3.34 | 59.82 |
| Path+Loc-3x | `path_loc_3x` | ✓ | | ✓ | 1 | 3 | 0.2147 | 3.08 | 59.96 |
| Interaction-2x | `interaction_2x` | ✓ | ✓ | ✓ | 1 | 2 | 0.5361 | 2.06 | 60.42 |
| Interaction-3x | `interaction_3x` | ✓ | ✓ | ✓ | 1 | 3 | 0.5361 | 1.94 | 60.57 |

说明：
- 论文正文 `paper/appme1.tex:1215-1217` 描述的设计取舍（Legacy 多档 vs 二值 3x、Path+Loc 更窄、
  Interaction 更宽、最终采用 v4 二值设计）**不构成本仓库的推荐**；六份 config / 六个 launcher 完全平行。
- 论文表中 Interaction-3x 一行有 `\rowcolor{gray!20}` 排印（`paper/appme1.tex:1197`），
  那是论文自身的排版；本仓库不据此设优先级，六变体并列。
- 表头 `Coverage` 指"被赋予强调权重的时空 latent 格占比"（`paper/appme1.tex:1101-1103`），
  MLR/IF 均为百分数，MLR 越低越好、IF 越高越好。

## 2. 六变体定义（本仓库权威定义在 `t4g_weightmap_variants.py:71-128`）

| 变体名 | 档位集 `levels` | base | 纳入加权的区域组 | 缓存目录 |
|---|---|---|---|---|
| `legacy_multilevel` | (0.5, 2, 3, 4, 6) | 0.5 | obj=4 / trans=6 / empty=2 / b_dest=3 / grip=3 / distractor=2 | `<root>/weightmap_cache_legacy_multilevel` |
| `binary_3x` | (1, 3) | 1 | obj=3 / trans=3 / empty=3 / b_dest=3 / grip=3 / distractor=3 | `<root>/weightmap_cache_binary_3x` |
| `path_loc_2x` | (1, 2) | 1 | obj=2 / trans=2 / b_dest=2 / empty=2 | `<root>/weightmap_cache_path_loc_2x` |
| `path_loc_3x` | (1, 3) | 1 | obj=3 / trans=3 / b_dest=3 / empty=3 | `<root>/weightmap_cache_path_loc_3x` |
| `interaction_2x` | (1, 2) | 1 | obj=2 / trans=2 / b_dest=2 / empty=2 / grip=2 | `<root>/weightmap_cache_interaction_2x` |
| `interaction_3x` | (1, 3) | 1 | obj=3 / trans=3 / b_dest=3 / empty=3 / grip=3 | `<root>/weightmap_cache_interaction_3x` |

- 未被变体纳入的区域组落 `base` 档（二值变体为 1x）；`bg` 永远是 base。
- `<root>` 的取值优先级：CLI `--root` > 环境变量 `T4G_WMAP_ROOT` > 默认 `PROBE`
  （= `/data/datasets/gagi/eve_v2_outputs/track4gen_probe`，见 `t4g_weightmap_variants.py:48-57,150-153`）。
- 六变体的级别集、区域落值、归一化、legacy 向后兼容、config 平行性由 `--self-test` 逐项断言
  （`t4g_weightmap_variants.py:382-512`）。
- `_check_registry()`（`t4g_weightmap_variants.py:356-371`）强制：
  `derived{region_levels} ∪ {base} == levels`、区域组名合法、
  legacy 的逐区域档位必须等于 `t4g_weightmap.REGION_VALUES`、base 必须等于 `W_BG`。
  改注册表时这些断言会挡住手误。

## 3. 论文列 -> 区域组 的映射

区域分解来自 `t4g_weightmap.build_weightmap_regions`（`t4g_weightmap.py`，自 `build_weightmap` 抽出，
二者对同一 anno 的输出逐位一致）。七个区域组：`obj`（物体轨迹）/ `trans`（源抓取窗与目的地放置窗）/
`b_dest`（两处"本该空"）/ `grip`（夹爪框）/ `empty`（空置前景）/ `distractor`（干扰物）/ `bg`（背景）。

论文三列按**语义**还原为区域组（定义见 `t4g_weightmap_variants.py:61-67`）：

| 论文列 | 本仓库区域组 | 依据 |
|---|---|---|
| Path/Loc. | `obj` + `trans` + `b_dest` + `empty` | `paper/appme1.tex:1098-1100`："restricts supervision to the target trajectory and source/destination regions" |
| Arm/Grip. | `grip` | 同上："additionally covers robot-arm, gripper, ..."；legacy 分解里只有夹爪框，没有手臂走廊标注 |
| Paste | `trans`（离线掩码）+ 训练时 `t4g_w_paste` clamp 抬到 3x（"实际贴入区"） | `paper/appme1.tex:1025` 掩码"additionally includes the actual pasted-instance region"；`paper/appme1.tex:1085` v4 restoration region 与实际贴入区各拿 3x |
| （无列）| `distractor` 单独一组 | 仅 `legacy_multilevel` 与 `binary_3x` 纳入（Path/Loc. / Interaction 两列在论文里不带干扰物） |

**判定说明（重要）**：论文表中 Path/Loc. 与 Paste 两列在**六行都是勾选**，所以"抓取窗/放置窗"
这一块区域同时被两列覆盖；本仓库把 `trans` 归入 Path/Loc.（六变体都开），
而 Paste 列那部分**离线掩码里不存在**的信息（"实际贴入区"要到训练时按 batch 里的
`paste_lat` 才知道）由 `t4g_joint_trainer.py:138-140` 的 `clamp_(min=t4g_w_paste)` 在 loss 内补足，
六变体统一 `t4g_w_paste=3.0`（`t4g_weightmap_variants.py:59,260`）。

## 4. 用法

### 4.1 预计算缓存（每次训练前一次，纯查表，无 GPU / 无检测）

```bash
# 六个变体各写一份缓存 (默认根目录: $T4G_WMAP_ROOT, 否则 /data/datasets/gagi/eve_v2_outputs/track4gen_probe)
python3 eveworld/pipeline/t4g_weightmap_variants_precompute.py

# 只写一个变体
python3 eveworld/pipeline/t4g_weightmap_variants_precompute.py --variant interaction_3x

# 覆盖缓存根目录 / 只跑部分 vid
T4G_WMAP_ROOT=/tmp/wmap python3 eveworld/pipeline/t4g_weightmap_variants_precompute.py --vids 1,2

# 自检 (合成假 anno, 不需要数据集) / 列出六变体
python3 eveworld/pipeline/t4g_weightmap_variants.py --self-test
python3 eveworld/pipeline/t4g_weightmap_variants.py --list-variants
```

其他参数：`--out-dir`（单个变体的精确输出目录，只配合单个 `--variant`）、
`--normalize`（**仅离线分析**，见第 5 节）、`--anno-dir` / `--assets-dir` / `--gripper-dir`。
与 `t4g_weightmap_precompute.py` 的关系：那个脚本写现行 legacy 缓存，行为不变；本脚本按变体**分目录**写六份，互不影响。

### 4.2 训练：每变体一份 config + 一个提交脚本

| 变体 | config 模块 | 提交脚本 | 产物根目录 |
|---|---|---|---|
| `legacy_multilevel` | `eveworld.pipeline.t4g_weightmap_legacy_multilevel_config` | `t4g_weightmap_legacy_multilevel_launch.sh` | `/data/datasets/gagi/eve_v2_outputs/t4g_weightmap_variants/legacy_multilevel` |
| `binary_3x` | ...`t4g_weightmap_binary_3x_config` | `t4g_weightmap_binary_3x_launch.sh` | ...`/t4g_weightmap_variants/binary_3x` |
| `path_loc_2x` | ...`t4g_weightmap_path_loc_2x_config` | `t4g_weightmap_path_loc_2x_launch.sh` | ...`/t4g_weightmap_variants/path_loc_2x` |
| `path_loc_3x` | ...`t4g_weightmap_path_loc_3x_config` | `t4g_weightmap_path_loc_3x_launch.sh` | ...`/t4g_weightmap_variants/path_loc_3x` |
| `interaction_2x` | ...`t4g_weightmap_interaction_2x_config` | `t4g_weightmap_interaction_2x_launch.sh` | ...`/t4g_weightmap_variants/interaction_2x` |
| `interaction_3x` | ...`t4g_weightmap_interaction_3x_config` | `t4g_weightmap_interaction_3x_launch.sh` | ...`/t4g_weightmap_variants/interaction_3x` |

六份 config 都是 9 行薄壳（`config = variant_config('<变体名>')`），
字段全部由注册表派生（`variant_config()`，`t4g_weightmap_variants.py:191-282`），
避免六份手写 config 漂移；**除权重图相关字段外，六份 config 的结构完全相同**（自检第 6 段断言）。
各 launcher：无参数打印计划；`check` 做静态检查 + `--self-test` + 缓存体检（**不训练**）；
`submit` 先做 `check`，发现 `<产物>/experiments/models/checkpoint*` 就拒绝续跑（rc=1），
否则经 `benchmarks/dreamgenbench/launch_gr1_train_kjob.sh` 提交（**提交前请确认 GAGI 路径与集群环境**）。

总入口 `t4g_weightmap_variant_matrix.sh`：`list`（列六变体级别/区域/缓存目录）、
`check`（依次跑六个 launcher 的 check）、`precompute`（六个变体各写一份缓存）；
训练仍逐个变体自行 `submit`（六者并列，无推荐）。

## 5. 与训练侧的契约（改权重图前必读）

1. **写盘存"档位原值"**：`precompute_variant(..., normalize=False)`（默认）把
   `base` / 各区域档位原样写进 `.npy`，并在写盘前断言 `np.unique(w) ⊆ levels`
   （`t4g_weightmap_variants.py:292-333`）。
2. **transform 侧校验**：`t4g_aug_trainer.py:118-130` 对每个 vid 检查
   `shape == (24, 30, 48)`、`np.isfinite`、`np.unique(w) ⊆ wmap_values`
   （config 里的 `wmap_values` 就是该变体的级别集 CSV）。
3. **trainer 侧账本**：`t4g_joint_trainer.py:128-132` 把 (24,30,48) 权重图上采样 2x 并断言尺寸；
   `t4g_joint_trainer.py:152-164` 在**逐样本归一化之前**记录档位图 `region_map`，
   再按 `|region_map - lev| < 1e-3` 做分区账本 —— 因此档位值必须精确等于 config 里声明的集合。
4. **loss 内归一化**：`t4g_joint_trainer.py:154` 逐样本除以自身均值（均值 1），
   所以训练只看**相对档位比**（如 3x vs 1x），与写盘尺度无关；
   不必（也不应）离线预归一化。
5. **贴入区抬权**：`t4g_joint_trainer.py:138-140` 对 `paste_lat` 区域做
   `clamp_(min=t4g_w_paste)`，即"不降低合同已给的高权重，只把该区抬到至少 `w_paste`"。
   六变体统一 `t4g_w_paste=3.0`（隔离"权重图"这一个变量）。
6. **`--normalize` 只用于离线分析/可视化**：它写出的图不满足第 2/3 条的契约，
   CLI 会打印警告；训练前请重新以默认（非归一化）模式预计算。

## 6. 覆盖率对照说明

- CLI 体检打印的 `覆盖率 (w > base)` 是**真实 anno 语义下重算**的强调格占比，
  与论文 Coverage 列同义但**不保证与作者的内部实现逐位相同**（我们的区域分解来自 `t4g_anno`，
  论文的 Coverage 来自其内部标注版本）。请把它当"同量纲的参考值"，不是复现值。
- `legacy_multilevel` 与 `binary_3x` 的覆盖率必然相等：两者覆盖的区域组集合相同，只是档位不同。
- 论文中 Interaction 系列的 Coverage（0.5361）显著高于 Legacy（0.2679），
  因为其掩码包含**手臂运动走廊**；本仓库 legacy 分解里没有手臂走廊标注这一项，
  对应信息在 `t4g_weightmap_armfix.py:50`（`ARMFIX_VERSION = "v4"`）/
  缓存 `weightmap_cache_uniform3_armfix_v4_nohuman_v2`（v4 arm-fix 路线）。
  所以本仓库按语义还原的 interaction 变体会**低于** 0.5361，这是预期差异，不是 bug。
- `--self-test` 打印的覆盖率基于**合成假 anno**（`_fake_anno()`，`t4g_weightmap_variants.py:374-379`），
  只用于验证六变体之间的相对关系，**不对应论文任何数字**。

## 7. 判定与不确定处（明示）

1. **`trans` 归 Path/Loc.**：依据是论文 Path/Loc. 与 Paste 两列在六行都勾选，
   且 `paper/appme1.tex:1098-1100` 把 Path+Loc 明确定义为
   "target trajectory and source/destination regions"。若作者原意是"Paste 列 = 抓放窗、
   不算 Path/Loc."，则 `path_loc_*` / `interaction_*` 四个变体应把 `trans` 移出
   （注册表里 `regions` / `region_levels` 各改一处即可，`--self-test` 会兜住一致性）。
2. **`t4g_w_paste` 取 3.0**：`t4g_joint_config.py:69` 历史值是 4.0，
   `t4g_joint_cleanv2_config.py:66` 是 3.0；这里取 3.0 以对齐论文
   "restoration-related locations 与实际贴入区同为 3x"（`paper/appme1.tex:1085,1217`）。
   历史 config 未被改动。
3. **`t4g_expected_samples=92`**：与 launcher 的 `EXPECTED_SAMPLES=92` 一致；
   `t4g_joint_cleanv2_config.py:64` 写的是 91（clean-v2 数据子集的历史值）。历史 config 未被改动。
4. **`distractor` 只进 legacy/binary_3x**：论文表中 Path/Loc. 与 Interaction 的定义不含干扰物；
   若想让 `binary_3x` 之外的变体也带干扰物，需显式改注册表（同样会被自检约束覆盖到）。
5. **legacy 逐位一致**：`legacy_multilevel` 的 `region_levels` 必须等于 `t4g_weightmap.REGION_VALUES`，
   base 等于 `W_BG`，且 `build_variant_weightmap(anno, 'legacy_multilevel', normalize=False)`
   必须与 `t4g_weightmap.build_weightmap(anno)[0]` 逐位相同（自检第 3 段）。
   改动 `t4g_weightmap.py` 的常量时，这里会立刻失败 —— 这是刻意设计的护栏。

## 8. 文件清单

| 文件 | 作用 |
|---|---|
| `t4g_weightmap_variants.py` | 六变体注册表 + 生成/预计算/自检/列表 CLI（本文件的核心） |
| `t4g_weightmap_variants_precompute.py` | 六变体缓存预计算入口（纯查表，无 GPU） |
| `t4g_weightmap_<variant>_config.py` | 六份训练 config 薄壳（字段由注册表派生，结构平行） |
| `t4g_weightmap_<variant>_launch.sh` | 六份提交脚本：打印计划 / `check` / `submit` |
| `t4g_weightmap_variant_matrix.sh` | 总入口：`list` / `check` / `precompute` |
| `t4g_weightmap.py` | 原有区域分解与本文件共用的底层（`build_weightmap_regions`）；`build_weightmap` 行为不变 |
