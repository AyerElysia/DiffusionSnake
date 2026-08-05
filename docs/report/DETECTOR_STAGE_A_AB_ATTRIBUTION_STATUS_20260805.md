# Detector Stage A：full-38 A→B 质量归因

更新时间：2026-08-05（Asia/Shanghai）  
负责人任务：`019fb3d5-abc9-7662-8731-8b8cb0c44755`  
状态：**一次性 full-38 A→B 评估完成；未训练、未调参、未改缓存/匹配/阈值/科学代码；C 未运行**

## 1. 结论

在冻结 Dense-6+H1、2×4 AB2=8-NFE、memory-off、seed `20260731`、full-38 显式 case-list、`network_input_pixels` V2 cache/adapter 下，A 和 B 均完整运行了 38 volumes / 6160 slices / 13,233 signed matched rows。

结论很明确：**当前 detector 对端到端质量的主要损失来自 significant-instance 覆盖不足，其次是 matched box 的定位几何。**

- D→A common-noise coverage：mean per-volume Dice 下降 `0.1292871`，NSD@2 下降 `0.1767488`；
- A→B geometry（oracle GT class）：mean per-volume Dice 再下降 `0.0894028`，NSD@2 再下降 `0.0635778`；
- D→B coverage+geometry：mean per-volume Dice 总下降 `0.2189782`，NSD@2 总下降 `0.2405540`；
- 38/38 病例的 Dice 和 NSD 差值方向都为性能下降，没有负损失/反向改善病例；
- 10,000 次 volume-level paired bootstrap 的 Dice/NSD 95% CI 全部不跨 0。

因此不能把 detector-box 端到端下降归因于 Flow。当前 detector 先漏掉大量 Flow significant components；对成功覆盖的 13,233 行，box geometry 又造成第二层稳定损失。

## 2. 冻结条件与执行边界

- D：31,772 个完整 GT significant instances；此前 final Flow zero-control 已 exact。
- A：13,233 个已签名 IoU≥0.1 matched rows，GT geometry + GT class。
- B：同一 13,233 rows、同一顺序，detector geometry + oracle GT class。
- C：`blocked_no_registered_class_provider`，未运行。
- A 先运行并完整结束；B 随后无条件完整运行，没有依据 A 中间结果修改或停止 B。
- checkpoint SHA256：`5e28f12df357ec4d18fc9f0baf67b5a57655932a585b4ae1a0254d8449ecfc72`
- Flow manifest v1.1 SHA256：`c9ad7b8ffba2f3e2c5698a35a77f4b0b3c9fab23cd735d2241ae23aac2f55698`
- A cache SHA256：`7120f154360c5d35403c06878496f3ce8f09f652d286cd3faf42fd429b559ca2`
- B cache SHA256：`b87dd9b951340846cf9aedaf9d360cbe2df91424a1e69b9aabc1ccca61b14455`
- D cache SHA256：`3c41077608ed78e0466830307972ff99c8d0e135c6c0ce154560d96e711c7d2c`
- V2 release SHA256：`f2c691ff3fdcaf2517e8d237ec9339617c290725f0ad8a383fc63e17bd0bbe60`
- strict validation SHA256：`f7ae5d25a6986a991adf137cac6eefbd3a5c3df81aae195c8a129af8ed9135e2`
- evaluator SHA256：`77fac9ab9cce2af7c544f11a1978e33891a99109e0b738041da6b5d1bf76f9da`
- direct adapter SHA256：`b36b762a550558781072e9a314f4290733c3d9509609c344087a9a5897c4a25b`
- 共享 GPU 只用于质量评估；`timing_reportable=false`，不得引用 summary 中的速度或时间。

## 3. 三条件指标

表中 pooled Dice/IoU 是全 38 例前景体素汇总；Dice、NSD@2、HD95 是 38 例等权平均。

| 条件 | pooled Dice | pooled IoU | mean-volume Dice | mean NSD@2 | mean HD95 (mm) | ASD |
|---|---:|---:|---:|---:|---:|---|
| D，完整 GT | 0.7894501 | 0.6521417 | 0.7940407 | 0.8093753 | 3.3850462 | unavailable |
| A common-noise，覆盖主控制 | 0.6710790 | 0.5049804 | 0.6647537 | 0.6326265 | 11.1299842 | unavailable |
| A full，完整独立运行 | 0.6709676 | 0.5048542 | 0.6644653 | 0.6323991 | 11.1439910 | unavailable |
| B，detector geometry + oracle class | 0.5777825 | 0.4062547 | 0.5750625 | 0.5688213 | 10.8133933 | unavailable |

授权冻结的 `compute_stage_a_metrics.py` 提供 NSD@2 和 HD95，但没有 ASD 实现。为遵守“不改代码”，本次没有修改指标工具或另行定义 ASD，机器结果将 ASD 明确记录为 `unavailable_under_frozen_tool`，不得伪造数值。

### common-noise 口径

完整 A rerun 与从 D 输出按同一 13,233 个 `instance_id` 保留得到的 common-noise A 很接近，但并非逐值相同：

- A-full minus A-common mean-volume Dice：`−0.0002884`；
- NSD@2：`−0.0002274`；
- HD95：`+0.0140068 mm`。

为了满足 D→A 的同一 D 噪声/轮廓实现，coverage 主结果采用 D→A-common；D→A-full 作为敏感性结果完整保留。A 和 B 拥有相同 13,233 行与顺序，因此 geometry 主结果采用 A-full→B。

## 4. 损失分解

Dice/IoU/NSD 使用 `reference − degraded`，正值表示性能损失；HD95 使用 `degraded − reference`，正值表示距离恶化。机器 JSON 同时保留逐指标原始 `reference − degraded`，避免符号歧义。

| 因子 | pooled Dice drop | pooled IoU drop | mean-volume Dice drop | NSD@2 drop | HD95 increase (mm) |
|---|---:|---:|---:|---:|---:|
| D→A common-noise，coverage 主结果 | 0.1183710 | 0.1471613 | 0.1292871 | 0.1767488 | +7.7449380 |
| D→A full，coverage 敏感性 | 0.1184825 | 0.1472875 | 0.1295755 | 0.1769762 | +7.7589448 |
| A full→B，geometry | 0.0931851 | 0.0985995 | 0.0894028 | 0.0635778 | −0.3305978 |
| D→B，coverage+geometry | 0.2116676 | 0.2458870 | 0.2189782 | 0.2405540 | +7.4283470 |

B 的 mean HD95 比 A 略低，但 Dice 和 NSD 在 38/38 病例中都下降。这说明单一 HD95 聚合受远端表面尾部分布影响，不应据此否认 geometry 的整体负效应。

## 5. Paired bootstrap，10,000 次

bootstrap unit 是 volume；每次从同一 38 个 paired case 中有放回抽样，seed `20260731`，percentile 95% CI。它只描述当前 38-volume 配对差值的不确定性，**不是独立测试集泛化证明**。

| 比较 | 指标 | mean paired delta | 95% CI |
|---|---|---:|---:|
| D→A common | Dice | 0.1292871 | [0.1149273, 0.1455140] |
| D→A common | NSD@2 | 0.1767488 | [0.1655846, 0.1878123] |
| D→A full sensitivity | Dice | 0.1295755 | [0.1151395, 0.1459013] |
| D→A full sensitivity | NSD@2 | 0.1769762 | [0.1657204, 0.1880512] |
| A→B | Dice | 0.0894028 | [0.0777491, 0.1008195] |
| A→B | NSD@2 | 0.0635778 | [0.0543379, 0.0728154] |
| D→B | Dice | 0.2189782 | [0.2028316, 0.2359487] |
| D→B | NSD@2 | 0.2405540 | [0.2297036, 0.2515277] |

## 6. Worst/best 与负向病例

条件自身 Dice 极值：

- D：worst `sub-verse093` 0.7538039；best `sub-verse023` 0.8294723。
- A-full：worst `sub-verse242` 0.5567356；best `sub-verse073` 0.7245633。
- B：worst `sub-verse242` 0.4477150；best `sub-verse071` 0.6704774。

coverage Dice 损失最大：

1. `sub-verse400_split-verse090`：0.2437111
2. `sub-verse225`：0.2407700
3. `sub-verse242`：0.2357146
4. `sub-verse412_split-verse235`：0.2165020
5. `sub-verse205`：0.2126737

geometry Dice 损失最大：

1. `sub-verse125`：0.1540674
2. `sub-verse116`：0.1527585
3. `sub-verse400_split-verse155`：0.1414895
4. `sub-verse230`：0.1377536
5. `sub-verse276`：0.1335018

D→B 总 Dice 损失最大：

1. `sub-verse242`：0.3455127
2. `sub-verse230`：0.3224536
3. `sub-verse221`：0.3056523
4. `sub-verse400_split-verse090`：0.3040247
5. `sub-verse412_split-verse235`：0.2769085

coverage、geometry、combined 三个比较中，Dice 与 NSD 的 negative-delta case 数均为 `0/38`。完整 38 例逐卷指标与 paired delta 均在机器 JSON 的 `per_case` 和各 factor 的 `paired_rows` 中。

## 7. 已签名覆盖审计解释

权威 coverage audit：

`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_full38_cache_20260804/coverage_audit.json`  
SHA256 `ebd89d481e52373ef1df8d30cd100cd877140ae8eaf0649cb78fc8812a34e327`

总体：31,772 GT significant instances 中匹配 13,233，IoU≥0.1 recall `0.4164988`；matched box mean IoU `0.4919628`。

component rank：

- rank 0：12,444/20,506，recall `0.6068468`；
- rank 1：743/9,444，recall `0.0786743`；
- rank 2：44/1,670，recall `0.0263473`；
- rank 3：2/152，recall `0.0131579`。

raw contour area：

- 2–9：2/906，recall `0.0022075`；
- 10–49：131/5,437，recall `0.0240942`；
- 50–199：3,088/10,954，recall `0.2819062`；
- 200+：10,012/14,475，recall `0.6916753`。

slice position：

- edge：1,132/5,790，recall `0.1955095`；
- center：4,796/10,594，recall `0.4527091`；
- transition：7,305/15,388，recall `0.4747206`。

连续漏检段 4,822 个，平均 3.8447 slices，最大 41；2,859 条轨迹中 2,824 条至少有一次漏检。未匹配 FP 轨迹 1,518 条，长度≥3 的 117 条，最大 13 slices。

A/B 只演化已签名 matched rows，未匹配 detector FP 没有进入 Flow。因此本次能量化 coverage 与 matched-box geometry 损失，但不能量化 FP 轮廓对端到端结果的破坏；这不是完整部署性能上界。

## 8. 运行失败审计

首次 A launcher 在模型/数据前因 worktree CWD 下相对 displacement-stats 路径不存在而停止；没有处理病例、没有生成质量结果，B 未启动。原日志保留：

`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_ab_full38_v2_20260805/a.log`  
SHA256 `27f15c83d74bc51801d6673d47460b0c6bda2765b1075e17b0e4611e91889ba2`

等价入口修复只把 CWD 切换到含冻结 stats 相对路径的主仓库；evaluator、checkpoint、config、cache、case-list、seed、顺序和科学参数均未改变。成功重试使用新 `_r1` 根目录，未覆盖失败目录。

## 9. 权威产物

正式根目录：

`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_ab_full38_v2_r1_20260805`

- 机器归因 JSON：`stage_a_ab_attribution_v2.json`  
  SHA256 `910705a8c49f2740bc806e2ca20b10d09c7d4fcc495c6050de5977a50bddf727`
- 冻结物理指标 JSON：`stage_a_metrics_v1.json`  
  SHA256 `608fa4af20983d4b706691cfb1facd28fd6e0a908c8f43f1394234c835b2f5b0`
- A summary SHA256 `411fc38b467eeb6cc842d1e7d56b33d345fbc1f269463f106f71fc44775846cb`
- B summary SHA256 `363f8b3082b4c015b982517da30c8df3d8d0e4c7167a5052547121767fd9c489`
- A instance contours SHA256 `954bcdf11b754c0198e4ed7ec0153ef137ce813f41072033663b9fb2eb5ca31e`
- B instance contours SHA256 `1a5fcbb3bb3e5e35bfad3d53bafce3fa8526f29889d5386f3d53072b572867c7`
- A log SHA256 `2ec0a92f8036d54eedd7bc6fd36469b2ac6c95915ca121a32f4f3e732f669fe9`
- B log SHA256 `dd58cb0c89f8d687d416916e1600ffeed19c1378a5f5aaefd76510558ffbd043`
- authoritative full inventory SHA：`FULL_INVENTORY_SHA256_V2.txt`  
  SHA256 `2f0b0a45d9a9ea67e1ca8f4935e554975acda90e8abc397d33ae2aa246014c23`，133 entries。
- authoritative size inventory：`FULL_INVENTORY_SIZE_BYTES_V3.tsv`  
  SHA256 `e4f39461017a017b53abef06f4e48c9797e9084507b267569168bcd0277e90ce`，133 entries。

首次 size inventory 因 shell 转义失败写成单行，文件保留审计但已被 V2 取代；其后 131-entry 中间库存又被包含最终 process/GPU audit 的 133-entry SHA V2 / size V3 取代。不得引用旧 `FULL_INVENTORY_SIZE_BYTES.tsv`、`FULL_INVENTORY_SHA256.txt` 或 `FULL_INVENTORY_SIZE_BYTES_V2.tsv` 作为最终库存。

## 10. 当前状态

- A/B 一次性质量归因已完成。
- C 未运行，继续 BLOCKED。
- 没有启动 detector/Flow 训练、阈值搜索、重新匹配或后续 Stage。
- 没有速度结论。
- 本任务在签发 final manifest 并确认无运行进程后 STOP。
