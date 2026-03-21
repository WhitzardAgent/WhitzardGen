# WhitzardGen

面向图像与视频合成数据采集的多模态生成框架。

英文版文档： [README.md](/Users/morinop/coding/whitzardgen/README.md)

这个仓库的目标不是做一个通用的模型服务系统，而是做一个**可运行、可追踪、可恢复、可导出**的数据生成框架，用统一 CLI 驱动不同的开源模型，最终产出适合下游使用的数据集。

## 项目定位

WhitzardGen 主要解决的是下面这类问题：

- 给一批 prompts
- 选择多个图像/视频模型
- 在单机多卡环境上高效跑推理
- 把结果组织成结构化 run 产物和 dataset bundle
- 支持中断恢复、失败重试和后续数据整理

当前重点是：

- 图像和视频生成
- persistent worker
- 多 replica 调度
- prompt 级别 ledger
- recovery
- 数据集导出与整理

## 当前能力

- Prompt 输入格式：
  - `.txt`
  - `.csv`
  - `.jsonl`
- Prompt 富字段支持：
  - `prompt_id`
  - `prompt`
  - `negative_prompt`
  - `parameters`
  - `metadata`
- Profile 运行：
  - `aigc run --profile ...`
- 运行期能力：
  - persistent worker
  - sequential replica warmup
  - multi-replica 执行
  - live throughput / ETA
  - `samples.jsonl`
  - `retry` / `resume`
  - failure policy
- 导出能力：
  - per-run `exports/dataset.jsonl`
  - dataset bundle
  - 多 run 合并导出
  - 按 split 组织
  - 按 model 过滤
  - `link` / `copy` 两种物化模式

## 目录说明

关键目录：

- [src/aigc](/Users/morinop/coding/whitzardgen/src/aigc)：框架源码
- [configs/models](/Users/morinop/coding/whitzardgen/configs/models)：按 `t2i` / `t2v` / `t2t` / `t2a` 拆分的模型注册表
- [configs/local_models](/Users/morinop/coding/whitzardgen/configs/local_models)：按任务类型拆分的本机/集群本地模型覆盖配置
- [configs/local_runtime.yaml](/Users/morinop/coding/whitzardgen/configs/local_runtime.yaml)：本地运行默认项
- [configs/run_profiles](/Users/morinop/coding/whitzardgen/configs/run_profiles)：可复用运行配置
- [prompts](/Users/morinop/coding/whitzardgen/prompts)：示例 prompt 文件
- [envs](/Users/morinop/coding/whitzardgen/envs)：每个模型对应的 env 规范与 validation 元数据
- [docs](/Users/morinop/coding/whitzardgen/docs)：架构和子系统文档
- [tests](/Users/morinop/coding/whitzardgen/tests)：轻量回归测试

## 安装

推荐：

```bash
pip install -r requirements.txt
```

可编辑安装：

```bash
pip install -e .
```

检查 CLI：

```bash
aigc version
```

如果你之前装过旧版本：

```bash
pip uninstall -y aigc
pip install -e .
```

如果 console script 仍然异常：

```bash
pip uninstall -y aigc
rm -rf *.egg-info
pip install --no-build-isolation -e .
```

## 环境管理方式

当前框架**不会**在 `aigc run` 时自动创建 Conda 环境。

现在的约定是：

- 由用户提前手工准备好 Conda env
- 框架负责解析模型对应 env 名称
- 检查环境是否存在
- 做轻量 validation
- 通过 `conda run -n <env_name> ...` 启动子进程

因此，真实运行前，建议先做：

```bash
aigc doctor
aigc doctor --model Z-Image
```

## 配置文件职责

当前推荐把配置理解为三层。

### 1. 模型定义层

[configs/models](/Users/morinop/coding/whitzardgen/configs/models)

这个目录定义“模型是什么”，当前按任务类型拆成：

- `configs/models/t2i.yaml`
- `configs/models/t2v.yaml`
- `configs/models/t2t.yaml`
- `configs/models/t2a.yaml`

每个分片里定义：

- adapter
- modality
- task type
- capabilities
- runtime 默认项
- Hugging Face repo 提示

它应尽量保持机器无关。

### 2. 本地覆盖层

[configs/local_models](/Users/morinop/coding/whitzardgen/configs/local_models)

这个目录定义“当前机器如何找到并运行它”，同样按任务类型拆成：

- `configs/local_models/t2i.yaml`
- `configs/local_models/t2v.yaml`
- `configs/local_models/t2t.yaml`
- `configs/local_models/t2a.yaml`

这些文件用来写：

- `conda_env_name`
- `local_path`
- `weights_path`
- `repo_path`
- `script_root`
- `hf_cache_dir`
- `max_gpus`

推荐理解方式：

- `configs/models/` 负责模型定义
- `configs/local_models/` 负责机器本地部署覆盖

示例：

```yaml
Z-Image:
  local_path: /models/Z-Image

Wan2.2-T2V-A14B-Diffusers:
  repo_path: /repos/Wan2.2
  weights_path: /models/Wan2.2-T2V-A14B-Diffusers
  max_gpus: 4

CogVideoX-5B:
  conda_env_name: cogvideo
  weights_path: /models/CogVideoX-5B
```

### 3. 本地运行默认项

[configs/local_runtime.yaml](/Users/morinop/coding/whitzardgen/configs/local_runtime.yaml)

主要控制：

- 默认 run 输出根目录
- 全局默认 seed

示例：

```yaml
paths:
  runs_root: /shared/aigc_runs

generation:
  default_seed: 12345
```

如果不设置 `generation.default_seed`，则默认保持随机生成，不会偷偷固定 seed。

## Prompt 格式

### TXT

最简单，一行一个 prompt：

```text
a futuristic city at night
a cat sitting on a chair
一只可爱的猫
```

### CSV

简单形式：

```csv
prompt
a futuristic city at night
```

或者带更多字段：

```csv
prompt_id,prompt,language,negative_prompt,parameters
p001,a futuristic city,en,"blurry","{""width"":1024}"
```

### JSONL

推荐实际采集任务使用 `.jsonl`，因为它最适合富字段。

示例：

```json
{"prompt_id":"p001","prompt":"a cinematic cat in warm morning light","negative_prompt":"blurry, low quality","parameters":{"width":1024,"height":1024,"guidance_scale":4.0},"metadata":{"split":"train","topic":"animals"}}
```

可参考：

- [prompts/example_image_rich.jsonl](/Users/morinop/coding/whitzardgen/prompts/example_image_rich.jsonl)
- [prompts/example_video_rich.jsonl](/Users/morinop/coding/whitzardgen/prompts/example_video_rich.jsonl)
- [prompts/canary_image.jsonl](/Users/morinop/coding/whitzardgen/prompts/canary_image.jsonl)
- [prompts/canary_video.jsonl](/Users/morinop/coding/whitzardgen/prompts/canary_video.jsonl)

## 生成参数优先级

当前参数优先级为：

```text
model defaults
< profile generation_defaults
< prompt-level parameters
```

意思是：

- model 自身有默认值
- profile 可以给一组 run 级默认参数
- 每条 prompt 的 `parameters` 可以再覆盖它们

示例 profile：

```yaml
generation_defaults:
  width: 1024
  height: 1024
  guidance_scale: 4.0
  num_inference_steps: 40
```

示例 prompt 覆盖：

```json
{"prompt_id":"p002","prompt":"a cat","parameters":{"width":1280}}
```

最终 `width=1280`。

## CLI 使用说明

### 模型查看

列出模型：

```bash
aigc models list
aigc models list --modality image
aigc models list --task-type t2v
```

查看单个模型：

```bash
aigc models inspect Z-Image
```

跑单模型 canary：

```bash
aigc models canary Z-Image --mock
aigc models canary Wan2.2-T2V-A14B-Diffusers
```

从当前 registry 生成 capability matrix：

```bash
aigc models matrix --write-docs
```

### 环境诊断

```bash
aigc doctor
aigc doctor --model Wan2.2-T2V-A14B-Diffusers
```

### 启动任务

单模型：

```bash
aigc run --models Z-Image --prompts prompts/canary_image.txt --execution-mode mock
```

多模型：

```bash
aigc run --models Z-Image,FLUX.1-dev --prompts prompts/canary_image.txt --execution-mode mock
```

真实视频任务示例：

```bash
aigc run --models Wan2.2-T2V-A14B-Diffusers --prompts prompts/canary_video.txt --execution-mode real
```

failure policy 示例：

```bash
aigc run \
  --models Z-Image \
  --prompts prompts/test_image_100.txt \
  --execution-mode real \
  --continue-on-error \
  --max-failures 20 \
  --max-failure-rate 0.10
```

### Run Profile

Profile 更适合真实采集：

```bash
aigc run --profile configs/run_profiles/image_real.yaml
aigc run --profile configs/run_profiles/video_real.yaml
```

可参考：

- [configs/run_profiles/image_real.yaml](/Users/morinop/coding/whitzardgen/configs/run_profiles/image_real.yaml)
- [configs/run_profiles/video_real.yaml](/Users/morinop/coding/whitzardgen/configs/run_profiles/video_real.yaml)

如果 CLI 参数和 profile 同时提供，CLI 参数优先。

### Run 查看

```bash
aigc runs list
aigc runs inspect <run_id>
aigc runs failures <run_id>
```

### Retry / Resume

重试失败输出：

```bash
aigc runs retry <run_id>
```

恢复中断任务中缺失的输出：

```bash
aigc runs resume <run_id>
```

### 数据集导出

单 run 导出：

```bash
aigc export dataset <run_id>
aigc export dataset <run_id> --mode link
aigc export dataset <run_id> --mode copy
```

多 run 合并导出：

```bash
aigc export dataset run_001 run_002 run_003
```

按 model 过滤：

```bash
aigc export dataset run_001 run_002 --model Z-Image --model FLUX.1-dev
```

自定义导出位置：

```bash
aigc export dataset run_001 run_002 --out /data/exports/my_bundle
```

## Run 产物说明

每个 run 通常会产生这些文件：

- `run_manifest.json`
- `failures.json`
- `samples.jsonl`
- `running.log`
- `runtime_status.json`
- `exports/dataset.jsonl`

它们的作用分别是：

- `run_manifest.json`：该 run 的总清单与 lineage
- `failures.json`：task 级失败摘要
- `samples.jsonl`：prompt 级 success/failure ledger
- `running.log`：详细时间戳日志
- `runtime_status.json`：运行中实时状态快照
- `exports/dataset.jsonl`：该 run 的 artifact-level JSONL 导出

## Export Bundle 说明

当前用户面向的数据集导出层会额外生成一个 bundle。

典型结构：

```text
dataset_bundle/
  dataset.jsonl
  export_manifest.json
  README.md
  media/
    train/
      Z-Image/
        image/
    val/
      Wan2.2-T2V-A14B-Diffusers/
        video/
    unspecified/
      ...
```

当前行为：

- 只包含成功且 artifact 存在的记录
- 支持多 run 合并
- 支持按 model 过滤
- 支持 `link` / `copy`
- 保留 source lineage 和原始 artifact 路径

## 当前模型模式

框架支持多种模型集成模式：

- diffusers in-process 图像模型
- diffusers in-process 视频模型
- repo-based Python in-process 模型
- external-process fallback 路径

目前 registry 中可见的代表模型包括：

- `Z-Image`
- `Z-Image-Turbo`
- `FLUX.1-dev`
- `stable-diffusion-xl-base-1.0`
- `Qwen-Image-2512`
- `HunyuanImage-3.0`
- `Wan2.2-T2V-A14B-Diffusers`
- `CogVideoX-5B`
- `LongCat-Video`
- `Wan2.2-TI2V-5B`
- `MOVA-720p`
- `HunyuanVideo-1.5`

实际请始终以：

```bash
aigc models list
```

为准。

## Adapter 结构

现在 adapter 已经按 modality 拆成 package，不再继续把所有实现都堆在一个大文件里。

当前结构：

- [src/aigc/adapters/images](/Users/morinop/coding/whitzardgen/src/aigc/adapters/images)
- [src/aigc/adapters/videos](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos)

例如：

- [src/aigc/adapters/images/zimage.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/images/zimage.py)
- [src/aigc/adapters/images/flux.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/images/flux.py)
- [src/aigc/adapters/videos/wan_t2v.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos/wan_t2v.py)
- [src/aigc/adapters/videos/cogvideox.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos/cogvideox.py)
- [src/aigc/adapters/videos/longcat.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos/longcat.py)
- [src/aigc/adapters/videos/helios.py](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos/helios.py)

这样共享逻辑会留在小的 base/common 模块里，后面接新模型也更容易维护。

## 如何适配一个新模型

建议按照下面顺序来做。

开始前也建议先看：

- [docs/model_integration_checklist.md](/Users/morinop/coding/whitzardgen/docs/model_integration_checklist.md)

### 1. 在 registry 中增加模型

修改 [configs/models](/Users/morinop/coding/whitzardgen/configs/models) 里对应的分片文件，补齐：

- `t2i.yaml`
- `t2v.yaml`
- `t2t.yaml`
- `t2a.yaml`

- `version`
- `adapter`
- `modality`
- `task_type`
- `capabilities`
- `runtime`
- `weights`

### 2. 在本地覆盖中补齐机器相关信息

如果需要，再修改 [configs/local_models](/Users/morinop/coding/whitzardgen/configs/local_models) 下对应分片：

- `conda_env_name`
- `local_path`
- `weights_path`
- `repo_path`
- `script_root`
- `hf_cache_dir`
- `max_gpus`

### 3. 手工准备 Conda 环境

当前策略是手工准备 env，而不是运行时自动创建。

准备好后先检查：

```bash
aigc doctor --model <model_name>
```

### 4. 实现或复用 Adapter

通常入口在：

- [src/aigc/adapters/images](/Users/morinop/coding/whitzardgen/src/aigc/adapters/images)
- [src/aigc/adapters/videos](/Users/morinop/coding/whitzardgen/src/aigc/adapters/videos)

按模型类型选择：

- diffusers in-process
- repo-based Python in-process
- external-process script

### 5. 添加验证和测试

建议至少补：

- registry 相关测试
- adapter 单测
- mock run-flow 测试
- doctor / env readiness 测试

### 6. 先做 canary，再 real run

先跑 dedicated canary：

```bash
aigc models canary <model_name> --mock
```

然后再做真实 GPU / cluster 验证：

```bash
aigc models canary <model_name>
```

建议一起看这些 onboarding 产物：

- [docs/model_integration_checklist.md](/Users/morinop/coding/whitzardgen/docs/model_integration_checklist.md)
- [docs/model_capability_matrix.md](/Users/morinop/coding/whitzardgen/docs/model_capability_matrix.md)
- [docs/model_capability_matrix.json](/Users/morinop/coding/whitzardgen/docs/model_capability_matrix.json)
- [configs/model_benchmarks.yaml](/Users/morinop/coding/whitzardgen/configs/model_benchmarks.yaml)
- [docs/model_benchmarks.md](/Users/morinop/coding/whitzardgen/docs/model_benchmarks.md)

## 一些模型的特别说明

### Wan2.2-T2V-A14B-Diffusers

这个模型通常涉及两类路径：

- `repo_path`：`Wan2.2` 仓库本地 checkout
- `weights_path`：`Wan-AI/Wan2.2-T2V-A14B-Diffusers` 的 Diffusers 权重目录

不要把 `weights_path` 指向非 Diffusers 结构的原始 checkpoint 目录。

### LongCat-Video

当前集成方式是 Python in-process，目标是：

- persistent worker 复用
- 一次加载 pipeline，多次批量推理

### CogVideoX-5B

当前是 in-process 且支持 replica-aware 运行。真实运行前建议先确认：

- `weights_path`
- `conda_env_name`
- `aigc doctor --model CogVideoX-5B`

## 实际使用建议

- 真实采集尽量使用 `.jsonl` prompts。
- 经常复用的任务尽量用 run profile。
- 开始真实任务前先跑 `aigc doctor`。
- 长任务中重点看：
  - `running.log`
  - `runtime_status.json`
  - `samples.jsonl`
- 中断恢复尽量使用：
  - `aigc runs retry`
  - `aigc runs resume`
- 面向下游数据整理时，优先使用 export bundle，而不是只拿单个 run 的 `exports/dataset.jsonl`。

## Roadmap

近期：

- 更丰富的 dataset card / export summary
- export-level dedupe / collision report
- 更完整的 train/val/test 数据组织
- 更强的数据质量审阅与过滤挂钩
- 更完善的真实集群回归验证

中期：

- annotation / review 流水线
- 基于 export bundle 的数据整理工具
- 更丰富的 artifact 统计与分析
- 更多模型与 adapter 的稳定化

长期：

- audio / text 模型支持
- 更大规模集群调度
- 下游评测 / 标注 / 安全分析整合
- 更完整的数据生命周期工具链

## 更多文档

如果你要看更底层的架构说明，继续读：

- [docs/spec.md](/Users/morinop/coding/whitzardgen/docs/spec.md)
- [docs/runtime_spec.md](/Users/morinop/coding/whitzardgen/docs/runtime_spec.md)
- [docs/cli_spec.md](/Users/morinop/coding/whitzardgen/docs/cli_spec.md)
- [docs/dataset_schema.md](/Users/morinop/coding/whitzardgen/docs/dataset_schema.md)
- [docs/model_registry_spec.md](/Users/morinop/coding/whitzardgen/docs/model_registry_spec.md)
- [docs/model_integration_checklist.md](/Users/morinop/coding/whitzardgen/docs/model_integration_checklist.md)
- [docs/model_capability_matrix.md](/Users/morinop/coding/whitzardgen/docs/model_capability_matrix.md)
- [docs/model_benchmarks.md](/Users/morinop/coding/whitzardgen/docs/model_benchmarks.md)
- [docs/prompt_spec.md](/Users/morinop/coding/whitzardgen/docs/prompt_spec.md)
- [docs/codex_tasks.md](/Users/morinop/coding/whitzardgen/docs/codex_tasks.md)
