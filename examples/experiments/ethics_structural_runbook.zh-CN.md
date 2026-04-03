# 伦理冲突评测操作手册

这份手册说明如何基于 WhitzardGen 当前的 benchmark core，运行一套完整的结构化伦理冲突评测实验。

适用对象：

- 想用现成的 ethics sandbox 模板包生成 benchmark
- 想批量测试一个或多个大语言模型
- 想对输出做 ethics-specific normalizer、judge 和 comparative analysis

当前示例依赖这些 example 组件：

- benchmark builder:
  - [examples/benchmarks/ethics_sandbox](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox)
- experiment recipe:
  - [examples/experiments/ethics_structural.yaml](/Users/morinop/coding/whitzardgen/examples/experiments/ethics_structural.yaml)
- ethics normalizer:
  - [examples/normalizers/ethics_structural](/Users/morinop/coding/whitzardgen/examples/normalizers/ethics_structural)
- ethics analysis plugins:
  - [examples/analysis_plugins/ethics_family_consistency](/Users/morinop/coding/whitzardgen/examples/analysis_plugins/ethics_family_consistency)
  - [examples/analysis_plugins/ethics_slot_sensitivity](/Users/morinop/coding/whitzardgen/examples/analysis_plugins/ethics_slot_sensitivity)

## 1. 你需要先准备什么

### 1.1 安装项目

```bash
pip install -e .
```

### 1.2 准备 target model 对应环境

当前 recipe 默认 target model 是 `Qwen3-32B`，所以你至少需要：

- 在 registry 中已有 `Qwen3-32B`
- 对应 conda 环境已准备好
- `whitzard doctor --model Qwen3-32B` 通过

建议先检查：

```bash
whitzard version
whitzard doctor --model Qwen3-32B
whitzard models inspect Qwen3-32B
```

### 1.3 确认 ethics sandbox 模板包存在

默认示例包位置：

- [package](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/package)

旧路径仍然保留兼容 alias：

- [docs/ethics_design/sandbox_template](/Users/morinop/coding/whitzardgen/docs/ethics_design/sandbox_template)

这个包至少应包含：

- `manifest.yaml`
- `slot_library.yaml`
- `analysis_codebook.yaml`
- `templates/*.yaml`

## 2. 先理解当前实验 recipe

当前默认 recipe 在：

- [examples/experiments/ethics_structural.yaml](/Users/morinop/coding/whitzardgen/examples/experiments/ethics_structural.yaml)

它定义了：

- benchmark builder：`ethics_sandbox`
- source package：`examples/benchmarks/ethics_sandbox/package`
- builder config：`examples/benchmarks/ethics_sandbox/example_build.yaml`
- synthesis model：`Qwen3-32B`
- target model：`Qwen3-32B`
- normalizer：`ethics_structural_normalizer`
- evaluator：`ethics_structural_judge`
- analysis plugins：
  - `ethics_family_consistency`
  - `ethics_slot_sensitivity`

对应的 builder config 是：

- [examples/benchmarks/ethics_sandbox/example_build.yaml](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/example_build.yaml)

现在它不再只是写一个 `realizations_per_template`，而是显式区分：

```yaml
sampling:
  realizations_per_template: 2

profiles:
  default_template_name: standard_naturalistic_v1

synthesis:
  model: Qwen3-32B

validator:
  enabled: true
  template_name: realization_validator_v1
  model: Qwen3-32B

validation:
  max_attempts: 2
```

也就是：

- 每个模板默认生成 2 个 realization
- 用 `standard_naturalistic_v1` writer prompt 生成第二人称、沉浸式的自然场景
- 每个场景都会同时产出结构化 `decision_options`，固定为 A/B 两个选择支
- 用 `realization_validator_v1` validator prompt 做 benchmark-feel / 冲突保真 / 二选一结构 / 第二人称沉浸感校验
- 用 `Qwen3-32B` 做 benchmark build 阶段的语义合成
- 合成失败时最多重试 2 次

如果你想关闭基于模型的 validator，只保留 deterministic guard，可以这样改：

```yaml
validator:
  enabled: false
```

## 3. 最简单的运行方式

直接使用 recipe：

```bash
whitzard evaluate run \
  --recipe examples/experiments/ethics_structural.yaml
```

这条命令会自动做这些事情：

1. 用 `ethics_sandbox` builder 采样 slot assignments
2. 用 writer prompt + synthesis model 把结构化 spec 实现成自然场景
3. 为每个场景生成结构化 A/B 两个选择支，供后续测试时按需拼接
4. 用 validator prompt 判断是否过于像 benchmark、是否软化冲突、是否破坏 binary framing、是否丢失第二人称沉浸感，并在需要时重试
5. 构建 benchmark
6. 用 target model 执行这些 benchmark cases
7. 用 `ethics_structural_normalizer` 先做规范化
8. 跑 record-level evaluator
9. 跑 group analysis 和 analysis plugins
10. 写出完整 experiment bundle

## 4. 分步运行方式

如果你想把 benchmark build 和 evaluate 分开，建议这样做。

### 4.1 单独构建 benchmark

```bash
whitzard benchmark build \
  --builder ethics_sandbox \
  --source examples/benchmarks/ethics_sandbox/package \
  --config examples/benchmarks/ethics_sandbox/example_build.yaml \
  --synthesis-model Qwen3-32B \
  --build-mode matrix
```

构建后可以检查：

```bash
whitzard benchmark inspect runs/benchmarks/<benchmark_bundle_dir>
```

这里的 `benchmark_bundle_dir` 就是完整 benchmark 的目录，不需要再导数据。它至少包含：

- `cases.jsonl`
  - 每一条 case 都已经带有完整 `slot_assignments`、`decision_frame`、`decision_options` 和其他 metadata
- `benchmark_manifest.json`
- `stats.json`
- `raw_realizations.jsonl`
  - build 阶段所有 realization 的原始结果和 validation 状态
- `rejected_realizations.jsonl`
  - 没进入最终 benchmark 的坏样本，但不会丢失

如果你关闭 model-based validator 或者允许部分坏样本存在，当前 build 也不会整批失败：

- valid cases 会写进 `cases.jsonl`
- invalid cases 会写进 `rejected_realizations.jsonl`
- 所以前面已经生成出来的 realization 不会白跑

### 4.2 在构建好的 benchmark 上跑 experiment

```bash
whitzard evaluate run \
  --benchmark runs/benchmarks/<benchmark_bundle_dir> \
  --targets Qwen3-32B \
  --normalizers ethics_structural_normalizer \
  --evaluators ethics_structural_judge \
  --analysis-plugins ethics_family_consistency ethics_slot_sensitivity
```

如果你想使用 evaluator 的 legacy flags，也可以继续显式指定：

```bash
whitzard evaluate run \
  --benchmark runs/benchmarks/<benchmark_bundle_dir> \
  --targets Qwen3-32B \
  --normalizers ethics_structural_normalizer \
  --evaluator-model Qwen3-32B \
  --evaluator-profile ethics_structural_review \
  --evaluator-template ethics_structural_review_v1 \
  --analysis-plugins ethics_family_consistency ethics_slot_sensitivity
```

## 5. 如何替换 target models

如果你想测多个模型，只需要改 recipe 的 `targets`，或者直接在命令行覆盖：

```bash
whitzard evaluate run \
  --recipe examples/experiments/ethics_structural.yaml \
  --targets Qwen3-32B Another-LLM
```

建议注意：

- 所有 target models 最好都是 `t2t`
- 对每个 target model 先跑 `whitzard doctor --model <model_name>`
- 如果 judge model 和 target model 不是同一个，也要确认 judge model 环境可用

## 6. 如何调 benchmark 规模

### 6.1 调每个模板生成多少个变体

修改：

- [examples/benchmarks/ethics_sandbox/example_build.yaml](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/example_build.yaml)

例如：

```yaml
sampling:
  realizations_per_template: 8
```

### 6.2 修改模板包本身

如果你要改伦理冲突结构、slot library、analysis codebook，请修改：

- [docs/ethics_design/sandbox_template](/Users/morinop/coding/whitzardgen/docs/ethics_design/sandbox_template)
- [package](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/package)

建议优先改这些内容：

- `templates/*.yaml`
- `slot_library.yaml`
- `analysis_codebook.yaml`

### 6.3 使用 `matrix` 模式

对于伦理冲突评测，建议继续使用：

```bash
--build-mode matrix
```

因为它更适合比较 sibling variants 的一致性和敏感性。

## 7. 输出结果在哪里看

experiment 完成后，会写出一个 experiment bundle。核心文件包括：

- `cases.jsonl`
- `target_results.jsonl`
- `normalized_results.jsonl`
- `score_records.jsonl`
- `group_analysis_records.jsonl`
- `analysis_plugin_results.jsonl`
- `experiment_manifest.json`
- `summary.json`
- `report.md`
- `failures.json`

你可以直接查看实验摘要：

```bash
whitzard experiments report <experiment_id>
```

或者 inspect：

```bash
whitzard evaluate inspect <experiment_id>
```

其中各文件含义建议这样理解：

- `cases.jsonl`
  - benchmark case 本体，保留 template/family/slot lineage
- `target_results.jsonl`
  - target model 的原始执行结果映射
- `normalized_results.jsonl`
  - ethics normalizer 提取后的 schema-light 中间层
- `evaluator_results.jsonl`
  - judge/rule evaluator 的结构化判断
- `group_analyses.jsonl`
  - core generic group aggregation 输出
- `analysis_plugin_results.jsonl`
  - ethics-specific comparative analysis 输出

## 8. 推荐的实际运行顺序

### 小规模冒烟

先把 `realizations_per_template` 设成 `1` 或 `2`，然后运行：

```bash
whitzard evaluate run --recipe examples/experiments/ethics_structural.yaml
```

重点看：

- benchmark 是否成功 build
- target model 是否能稳定跑完
- `normalized_results.jsonl` 是否提取出了 decision/refusal/reasoning
- `analysis_plugin_results.jsonl` 是否有 family consistency / slot sensitivity 输出

### 中规模实验

把 `realizations_per_template` 提到 `4` 到 `8`，再逐步增加 target models。

建议流程：

1. 先单模型
2. 再双模型对比
3. 最后再扩大 template 覆盖和 realization 数量

## 9. 常见修改点

### 修改 target models

改：

- [examples/experiments/ethics_structural.yaml](/Users/morinop/coding/whitzardgen/examples/experiments/ethics_structural.yaml)

### 修改 benchmark realization 强度

改：

- [examples/benchmarks/ethics_sandbox/example_build.yaml](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/example_build.yaml)

### 修改 ethics-specific normalizer / analysis logic

改：

- [examples/normalizers/ethics_structural/normalizer.py](/Users/morinop/coding/whitzardgen/examples/normalizers/ethics_structural/normalizer.py)
- [examples/analysis_plugins/ethics_family_consistency/plugin.py](/Users/morinop/coding/whitzardgen/examples/analysis_plugins/ethics_family_consistency/plugin.py)
- [examples/analysis_plugins/ethics_slot_sensitivity/plugin.py](/Users/morinop/coding/whitzardgen/examples/analysis_plugins/ethics_slot_sensitivity/plugin.py)

## 10. 当前这份手册的边界

这份 runbook 关注的是：

- 如何把 ethics benchmark 跑起来
- 如何理解当前 example package 和 generic core 的关系
- 如何定位 benchmark / experiment artifacts

它不覆盖：

- UI 展示
- 更复杂的自动 launcher/orchestrator
- 伦理研究维度本身的理论解释

如果你接下来要做更复杂的伦理研究扩展，建议优先沿着 `examples/` 继续加：

- 新的 benchmark package
- 新的 normalizer
- 新的 analysis plugin
- 新的 experiment recipe
