请基于我当前的 Part 3 代码库，实现一条**新的、独立的、可运行的预训练模型版 pipeline**，核心目标是：

- 使用 **SEA-RAFT** 作为 `C_temp / S_flow` 的 backend
- 使用 **MASt3R** 作为 `C_feat / S_feat` 的 backend
- 保留我当前已有的“手动 conf + consis 路线”不变
- 新路线与旧路线在代码和脚本上清晰区分，但可以复用公共模块
- 最终能够从 trajectory manifest / pseudo views 出发，离线预计算 component maps，再接入现有 hybrid scene 和 3DGS 训练/eval 体系

====================
一、总体要求
====================

请不要重写整个 Part 3 项目，而是做一条**并行的新路线**：

1. **保留旧路线**
   - 我现在已有的 manual confidence / consistency 路线必须继续可用
   - 现有脚本、配置、训练流程不要破坏

2. **新增一条预训练模型路线**
   - 这条路线专门用 MASt3R + SEA-RAFT 做 confidence / consistency backend
   - 最终输出仍然要兼容我现有的：
     - confidence manifest
     - hybrid scene
     - train_3dgs_confidence.py
     - compare/eval 脚本

3. **优先复用公共模块**
   - 如果 geometry / manifest / hybrid / eval / config 现有代码能复用，请复用
   - 但新旧路线的入口脚本、配置文件、backend 逻辑必须清楚分开

4. **不要把 MASt3R / SEA-RAFT 塞进训练 loop**
   - 新路线必须走“离线预计算 component maps -> 缓存 -> 训练时只读取”的方式
   - 训练期不要实时跑 MASt3R / SEA-RAFT

====================
二、参考官方仓库（请联网查看）
====================

请你联网查看下面两个仓库，并据此实现：

1. MASt3R
https://github.com/naver/mast3r

2. SEA-RAFT
https://github.com/princeton-vl/SEA-RAFT

实现时请优先参考官方推荐的推理方式、checkpoint 使用方式和最小依赖安装方式。

如果你发现有更合适的官方推理脚本或简化调用方式，请在实现总结里说明。

====================
三、我要的新路线目标
====================

这条新路线的目标不是替代 DynamiCrafter，不是重做 Part 2，也不是重做几何底座；而是：

- 用 **SEA-RAFT** 替换当前轻量 temporal backend，增强：
  - `C_temp`
  - `S_flow`

- 用 **MASt3R** 替换/增强当前 feature backend，增强：
  - `C_feat`
  - `S_feat`

- 必要时，给 pose refinement 增加一个 **MASt3R fallback**
  - 只在 ORB / PnP 匹配不足时启用
  - 不要一开始就重写整个 pose refinement 主链

====================
四、希望你实现的目录与结构
====================

请不要把所有新逻辑继续堆进原有文件里。请新增一套清晰的模块结构，例如：

- `part3/src/part3_stack_pretrained/`
  - `pipeline_pretrained.py`
  - `mast3r_backend.py`
  - `sea_raft_backend.py`
  - `feature_confidence.py`
  - `temporal_confidence.py`
  - `clip_consistency.py`
  - `pose_refine_mast3r.py`   （如果做 fallback）
  - `utils_pretrained.py`

- `part3/apps/`
  - `generate_confidence_pretrained.py`
  - `build_hybrid_pretrained.py`
  - `run_part3_pretrained.py`

- `part3/configs/`
  - `project_pretrained_re10k.json`
  - `project_pretrained_405841.json`

- `part3/scripts/`
  - `run_part3_pretrained.sh`
  - `eval_part3_pretrained.sh`

如果你认为更合适的命名方式不同，可以调整，但必须满足：
- 新路线和旧路线明确区分
- 入口脚本明确分开
- 公共逻辑复用，但 backend 逻辑独立

====================
五、功能要求：你要实现什么
====================

--------------------------------
[1] 新路线入口：pretrained pipeline
--------------------------------

请实现一条新的 pipeline 入口，大致流程如下：

1. 读取 trajectory manifest / pseudo views
2. 读取已有 pseudo images、pose、coarse render、reprojection metadata
3. 用 MASt3R 计算 `C_feat`
4. 用 SEA-RAFT 计算 `C_temp`
5. 结合现有几何逻辑与 reprojection 逻辑，生成：
   - `C_vis`
   - `C_reproj`
   - `C_feat`
   - `C_temp`
6. 进一步聚合成：
   - `S_flow`
   - `S_reproj`
   - `S_feat`
   - `S_pose`（如果有）
   - `clip_score`
7. 写出新的 confidence manifest 和 `.npy` component maps
8. 复用现有 hybrid scene 写回逻辑，构建训练目录
9. 复用现有 3DGS 训练/eval 脚本

换句话说：
- 训练主链尽量不改
- 重点替换/增强的是离线 confidence / consistency 预计算 backend

--------------------------------
[2] SEA-RAFT backend
--------------------------------

请实现一个可独立调用的 SEA-RAFT backend，用于：

- 输入：相邻 pseudo frames 或 pseudo vs neighbor frame
- 输出：
  - flow
  - uncertainty（如果官方可直接给出）
  - forward-backward consistency
  - 最终像素级 `C_temp`
  - clip 级 `S_flow`

要求：
- 输出格式要能兼容现有 confidence manifest/component map 的写法
- 如果 uncertainty 没有现成字段，就用 forward-backward flow consistency 生成 `C_temp`
- 不要直接替换原有 Farneback 代码，而是新增 backend：
  - 例如 `temporal.backend = "farneback" | "sea_raft"`

--------------------------------
[3] MASt3R backend
--------------------------------

请实现一个可独立调用的 MASt3R backend，用于：

- 输入：pseudo frame 和 anchor/coarse/reference frame
- 输出：
  - dense / reciprocal matching 质量
  - descriptor similarity
  - patch/region 级 feature confidence
  - 最终像素级或 patch 级 `C_feat`
  - clip 级 `S_feat`

要求：
- 优先使用 MASt3R 官方仓库里适合做推理和 reciprocal matching 的接口
- 不要跑完整 SfM / global alignment
- 先只做：
  - `desc`
  - reciprocal matching
  - match quality / match density / match confidence
- 用这些信息生成 `C_feat`

要求新增 backend：
- 例如 `feature.backend = "light" | "dust3r" | "mast3r"`

--------------------------------
[4] C_vis / C_reproj 复用现有逻辑
--------------------------------

请不要重做 `C_vis` 和 `C_reproj` 主体逻辑。

要求：
- 优先复用当前已有的：
  - coarse geometry render
  - depth / alpha / normal / padding validity
  - reprojection validity / reprojection error
- 新路线主要增强的是：
  - `C_feat`
  - `C_temp`
  - `S_feat`
  - `S_flow`

但是如果为了适配新 backend，需要在 confidence 融合上加少量接口，请最小化修改。

--------------------------------
[5] confidence 融合方式：新路线不要破坏现有训练接口
--------------------------------

新路线最终还是要输出和旧路线兼容的：

- final mask / alpha
- component maps
- clip score
- confidence manifest

建议最终仍然形成类似：

- `hard_validity`
- `soft_confidence`
- `final_mask`

并复用当前训练侧的 normalized masked loss。

注意：
- 训练链可尽量不动
- 重点是让新路线输出符合现有训练输入格式

--------------------------------
[6] 可选：MASt3R pose refinement fallback
--------------------------------

如果实现成本可控，请增加一个可选的 pose refinement fallback：

- 默认仍走现有 ORB + PnP + refine
- 当 ORB 匹配数 / inlier 数不足时，再尝试 MASt3R reciprocal matches
- 仍然复用现有 PnP / refine 主逻辑
- 不要一开始就重写 pose refinement 全流程

这个 fallback 必须是可配置开关，不要默认强制启用。

====================
六、配置要求
====================

请为新路线设计独立配置，例如：

- `temporal.backend = "sea_raft"`
- `feature.backend = "mast3r"`
- `pose_refine.fallback_backend = "mast3r"`
- `pretrained.cache = true`
- `pretrained.batch_size = ...`
- `pretrained.device = "cuda"`
- `pretrained.checkpoint_paths = {...}`

同时保留：
- 旧路线 config 不受影响
- 新路线 config 与旧路线字段尽量兼容，但通过 backend 名称区分

请分别给：
- Re10k 用的 config
- 405841 用的 config

如果你判断需要 dataset-specific config，请直接在新配置里体现。

====================
七、复用要求
====================

请尽量复用当前已有模块，例如：

- trajectory manifest 读取
- pseudo view / coarse render 路径组织
- confidence manifest 写出逻辑
- hybrid scene 组织
- train_3dgs_confidence.py
- compare_part3_metrics.py

不要重复造轮子。

但要求你清楚地把：
- 旧路线 backend
- 新路线 backend
分离出来。

====================
八、测试与验收要求
====================

请在实现后，给出一个最小可运行验证方案，至少包括：

1. 不重跑 DynamiCrafter，只基于现有 pseudo views 预计算一套：
   - `C_feat`（MASt3R）
   - `C_temp`（SEA-RAFT）
   - `clip_score`

2. 用这套新 confidence manifest 构建一个 hybrid scene

3. 复用现有 3DGS 训练和 eval 脚本，完成一轮：
   - Re10k 小规模验证
   - 405841 小规模验证

4. 输出对比建议：
   - 旧路线（manual conf/consis）
   - 新路线（pretrained backend）
   - 至少比较：
     - mean_feature_confidence
     - mean_temporal_confidence
     - clip_score
     - patch_keep_ratio（如果还在）
     - real-only PSNR/SSIM/LPIPS

====================
九、我希望你最后交付的内容
====================

请在完成代码修改后，输出一份实现总结，说明：

1. 新增了哪些文件
2. 修改了哪些旧文件
3. 新路线和旧路线分别怎么跑
4. 哪些逻辑被复用了
5. MASt3R 现在负责什么
6. SEA-RAFT 现在负责什么
7. pose refinement fallback 是否实现
8. 是否有未完成项 / 风险点
9. 给出建议的 first-run 命令

====================
十、风格要求
====================

- 请优先做“最小但完整可运行”的实现
- 不要把目标膨胀成第二套大系统
- 不要破坏旧路线
- 不要只给建议，请直接改代码
- 如果某一步实现风险过高，请保留接口并在总结里说明
- 优先保证：
  - backend 可插拔
  - manifest 可落盘
  - hybrid / train / eval 可串起来

请开始直接实现。