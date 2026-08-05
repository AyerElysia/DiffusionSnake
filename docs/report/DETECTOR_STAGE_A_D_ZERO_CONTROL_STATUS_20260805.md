# Detector Stage A：D-cache vs native GT 最终 Flow 零对照

更新时间：2026-08-05（Asia/Shanghai）  
负责人任务：`019fb3d5-abc9-7662-8731-8b8cb0c44755`  
状态：**formal full-38 D-only zero-control PASS；A/B/C 仍为 NO-GO；未训练 detector 或 Flow**

## 1. 正式结论

在冻结的 Dense-6+H1 Flow 上，`network_input_pixels` 条件 D cache 与 native H1 GT 输入完成了完整 38-volume 成对零对照：

- 38 volumes / 6160 slices / 31,772 instances；
- 最终 feature-space contour：`torch.equal=true`，`max_abs=0.0`，mismatch values `0`；
- 还原到图像空间的 contour：`torch.equal=true`，`max_abs=0.0`，mismatch values `0`；
- `py_ind`、`instance_id`、`flow_instance_order`、`flow_class_id` 全部 exact，mismatch count 均为 `0`；
- 38/38 病例的预测栅格、GT 栅格、slice indices、Dice 和 IoU 均 exact；
- 全量预测前景与 GT 前景 mismatch voxels 均为 `0`；
- cache-minus-native 的 processed slices、volume mean Dice、volume mean IoU 差值均为 `0`；
- 校验失败列表为空，机器状态为 `pass`。

这证明新的 `network_input_pixels` D-cache/provider 执行路径在最终 Flow 输出上与 native GT 路径**逐值等价**，坐标合同修复可以正式签认。它不证明 detector 质量提高，也不构成新的 Flow 质量结果。

## 2. 冻结协议与绑定

- Flow interface v1.1：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/flow_interface_freeze_20260804/FLOW_INTERFACE_FREEZE_MANIFEST_V1_1_20260804.json`  
  SHA256 `c9ad7b8ffba2f3e2c5698a35a77f4b0b3c9fab23cd735d2241ae23aac2f55698`
- `label_id=1..25`；`flow_class_id=label_id-1=0..24`；旧 v1.0 只保留审计。
- Flow checkpoint：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/output_head_h0_h1_h2_20260803/distilled/h1_distilled_full.pt`  
  SHA256 `5e28f12df357ec4d18fc9f0baf67b5a57655932a585b4ae1a0254d8449ecfc72`
- base config SHA256 `d57ff1f9e0b620022e173e767bb03cad20c9ee126779b491cd323f4fb78755ae`
- external-cache config SHA256 `a76db74ad0148301e9b05ce3847c1ba964db250eb3e34b07ea521c0d27e4b529`
- displacement stats SHA256 `847d133b4ab154bf8fe82e772eee94bd3bd87b7391a38a83611821e93b787ce7`
- 2 outer × 4 inner、AB2、8-NFE protocol label、memory-off、seed `20260731`；native/cache 使用同一 GPU、同病例、同噪声组织和同实例顺序。
- 物理 GPU 0 为共享 GPU；本门只用于数值等价性，`timing_reportable=false`，禁止引用 E2E 速度或延迟。

## 3. 前置 input-space 硬合同

V2 release：

- `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_full38_inputspace_cache_20260805/INPUTSPACE_CACHE_RELEASE_MANIFEST_V2_20260805.json`
- SHA256 `f2c691ff3fdcaf2517e8d237ec9339617c290725f0ad8a383fc63e17bd0bbe60`

full-38 strict validation：

- `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_full38_inputspace_cache_20260805/cache_contract_validation_inputspace_full38_strict_v2.json`
- SHA256 `f7ae5d25a6986a991adf137cac6eefbd3a5c3df81aae195c8a129af8ed9135e2`
- 6160 slices / 31,772 rows；bbox、score、class、shape、row order mismatch 均为 `0`；bbox 与 detection row `max_abs=0.0`；六组初始化张量全部 `torch.equal`。
- `sub-verse071` slices 186–197 的 12 行与 `sub-verse150` 的 10 行 bottom-edge 定点审计为 22/22 exact。

正式 D cache：

- `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_full38_inputspace_cache_20260805/condition_d_formal_full38_inputspace_full_gt_control.json`
- SHA256 `3c41077608ed78e0466830307972ff99c8d0e135c6c0ce154560d96e711c7d2c`

MoonViT feature cache 全量覆盖审计：

- `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_full38_inputspace_cache_20260805/formal_full38_moonvit_cache_coverage_20260805.json`
- SHA256 `f972aabb4f3453ea927116fe8903b488f0ee57b0b9d013fafe39fa7e7b4f3ea0`
- 6160/6160 feature files present，状态 `pass`。

## 4. selected3 smoke（只作协议门，不晋级）

smoke D cache：

- `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_d_zero_control_smoke_cache_v2_20260805/condition_d_smoke_010_011_013_inputspace_full_gt_control.json`
- SHA256 `de2cd0c55b19a8e903fc6cd42285eaab0fc0addbb3bd3b2b903b72dd05b10bf1`

smoke strict contract：

- `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_d_zero_control_smoke_cache_v2_20260805/cache_contract_validation_smoke_d_strict_v2.json`
- SHA256 `d05b0a76e93f29778d0274eb2a7cc8b1735476de0dc2c7e138db9fec82c4454a`
- 333 slices / 1796 instances，detection rows 和初始化张量全部 exact。

最终 Flow smoke：

- 机器 JSON：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_d_zero_control_smoke_v2_r1_20260805/d_final_flow_zero_control.json`  
  SHA256 `4d56c6430cfc25a52847c5c6f3d024e5d87ae4983afd15e40d0d6e8a7c6f9be7`
- manifest：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_d_zero_control_smoke_v2_r1_20260805/D_ZERO_CONTROL_MANIFEST.json`  
  SHA256 `e6f7e4bfd6e9e8fd230f1650d28a9f858c9859af75ad6131eb71d7103cb93217`
- 333 slices / 1796 instances；final/restored contours、IDs/order/class、raster、Dice/IoU delta 全部 exact 0。
- smoke 状态固定为 `smoke_d_final_flow_zero_control_pass_not_eligible_for_promotion`。

smoke 成对运行的 native/cache absolute volume mean Dice 均为 `0.7940801009`。该数值不与 RL signed anchor `0.7944618785` 处在跨实验同一噪声/运行合同下，**不得横向比较、不得替换论文 Flow 基线**；smoke 只引用 delta `0` 和 exact 等价性。

## 5. full-38 正式零对照产物

正式根目录：

`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_d_zero_control_full38_v2_r1_20260805`

权威机器结果：

- JSON：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_d_zero_control_full38_v2_r1_20260805/d_final_flow_zero_control.json`  
  SHA256 `01e6ea8aeb5339baf8fa29a54f26f2a65817147a76b345377b7353cbe3bff033`
- manifest：`/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_d_zero_control_full38_v2_r1_20260805/D_ZERO_CONTROL_MANIFEST.json`  
  SHA256 `c0214e9bc0edfe11c0eeb03d17c0e1c365317a32f64ae64088b1148b2cf69b65`
- manifest 状态：`formal_full38_d_final_flow_zero_control_pass_a_b_still_no_go`。

成对 summary 与日志：

- native summary SHA256 `5a81ffeb9b8a21a18a905b97930914b618df5c6b8e599363e85bd960ba1340f2`
- cache summary SHA256 `2e57363c80e94fe6574dc048fa70472527a4d212604c5b49fe621cbd6756c703`
- native log SHA256 `18ac24bab7b964d00be6e02e4171766e8bbc5287455fc9e6def3520cf081edaa`
- cache log SHA256 `e20443324c7fe86a58d3fde6043ece3eb54360e1593f9df45594a55cb0323f8b`
- native/cache 的 instance-contour JSONL SHA 分别记录在权威机器 JSON 中。

full-38 成对机器 summary 中 native/cache absolute volume mean Dice 均为 `0.7940407262`。按论文统筹新增口径，本门**只引用 cache-minus-native delta=0 和 exact 等价性**；该绝对值不得替换论文 Flow 基线、不得升级为质量结果，也不得与其他未共享同一运行/噪声合同的绝对分数比较。

## 6. 保留的失败与重试审计

所有失败目录均保留，成功重试使用新目录，未覆盖旧产物：

1. 首次 smoke 在进入推理前停止：worktree CWD 下相对路径 `data/stats/volmem_sagittal_disp_stats.json` 不存在。失败日志：  
   `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_d_zero_control_smoke_v2_20260805/d_native_gt.log`  
   SHA256 `32cff352d7322cc99ad6430c0404fe1a7f4ca6b93b5267b8e726e4643db39ab7`
2. 首次 formal 在 native-D 约 200 slices 时停止：`/dev/shm` feature cache 缺少 `sub-verse016/x0000.npz`；cache-D 未启动。失败日志：  
   `/home/medteam/Zhrch/DiffusionSnake-12-30/data/outputs/volmem/diagnostics/detector_stage_a_d_zero_control_full38_v2_20260805/d_native_gt.log`  
   SHA256 `c8173b2ec707a18b44cc3bd6256a28db3e11810a11634bc0e4b85d459cfcab4b`
3. 重试只修复运行资源路径：使用绝对 displacement-stats 路径与经 6160/6160 覆盖审计的磁盘 MoonViT cache；没有修改 detector、Flow、D cache、病例、噪声、顺序或数值协议。

旧 original-image 坐标合同失败继续保留：

- FAIL JSON SHA256 `04ce44e2cdd43b614f41b467c1bc0339fdb0dab801e9d887842b371fee71e246`
- diagnostic v2 SHA256 `2318cd903f3d3ff125e956fa0322abfa4c6a7612f0507f92f98e04264a607b01`
- 22 个 bottom-edge 往返误差只归类为旧坐标合同失败，不是 Flow 或 detector 质量结论。

## 7. 实现与复现实证 SHA

- evaluator SHA256 `77fac9ab9cce2af7c544f11a1978e33891a99109e0b738041da6b5d1bf76f9da`
- strict zero-control validator SHA256 `a45b4fb98fb34662469f8fa8973f50aa5519f27dfa82447b2e278a576c89cea2`
- D-only runner SHA256 `8ca07eedac8eaed031d6a2ced0b6aa6ce18b67da79c8f3c3fbddf5809c0382a1`
- manifest writer SHA256 `362a9e9afd972f3b29fa55f730db49e1bdd21cb161dbff42ca1cb0fb6e543cdf`
- smoke cache builder SHA256 `8164b36bfaebf616404117ee9a021f738684dc03aa096e1601a1f702b06e9362`
- feature coverage validator SHA256 `90afc3d5b7b6b8abb0b72a7392eb3a947327db9bf9206c873c503d33fa9a870a`

## 8. 当前边界与下一门

- A/B/C 没有运行；没有启动 detector 或 Flow 训练。
- C 继续是 `blocked_no_registered_class_provider`。
- A/B 正式联评仍为 `NO-GO`，只有论文统筹或项目负责人另行放行后才能执行。
- 当前无本 Stage-A 的 GPU evaluator、runner 或训练进程。
- 本次完成的是缓存执行路径的零对照门，不改变既有 detector coverage 结论，也不把 detector 漏检/几何损失归因于 Flow。

