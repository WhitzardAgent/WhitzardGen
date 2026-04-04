# 当前架构下如何运行伦理冲突评测

这份文档说明，**在当前 Phase 38 的 V2 + semantic realization 架构下**，如何用 WhitzardGen 运行一套结构化伦理冲突评测。

这不是旧的“单纯 prompt -> run -> annotate”流程，而是：

```text
benchmark build
-> semantic realization build
-> EvalTask compile
-> target execution
-> normalization
-> scoring
-> group analysis
-> analysis plugins
-> experiment bundle
```

对应到当前项目里的心智模型是：

- `run engine`：底层执行内核
- `benchmark builder`：把伦理模板包变成 canonical benchmark cases
- `semantic realization build`：先采样 slot，再用 writer prompt 调用 T2T 模型生成第二人称自然场景和结构化 A/B 选择支，再用 validator prompt 做 build-time 质量判定与重试
- `normalizer`：把 target 输出规整成可比较的中间层
- `scorer`：做 rule/judge 级评分与抽取
- `group analyzer / analysis plugin`：做 family / sibling / slot 维度聚合分析

## 1. 你现在实际会用到哪些组件

当前伦理冲突评测示例主要由这些部分组成：

- ethics benchmark builder：
  - [builder.py](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/builder.py)
- ethics benchmark source package：
  - [package](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/package)
- ethics normalizer：
  - [normalizer.py](/Users/morinop/coding/whitzardgen/examples/normalizers/ethics_structural/normalizer.py)
- ethics analysis plugins：
  - [plugin.py](/Users/morinop/coding/whitzardgen/examples/analysis_plugins/ethics_family_consistency/plugin.py)
  - [plugin.py](/Users/morinop/coding/whitzardgen/examples/analysis_plugins/ethics_slot_sensitivity/plugin.py)
- ethics experiment recipe：
  - [ethics_structural.yaml](/Users/morinop/coding/whitzardgen/examples/experiments/ethics_structural.yaml)

如果你想先看更贴近 examples 的短版说明，也可以看：

- [ethics_structural_runbook.zh-CN.md](/Users/morinop/coding/whitzardgen/examples/experiments/ethics_structural_runbook.zh-CN.md)

这份 `docs/` 里的文档更强调**当前核心架构下应该怎么理解和操作**。

## 2. 运行前先确认什么

### 2.1 target model 可用

默认示例 recipe 用的是：

- `Qwen3-32B`

所以至少先确认：

```bash
whitzard doctor --model Qwen3-32B
whitzard models inspect Qwen3-32B
```

如果你要换成远程 OpenAI-compatible judge 或别的 target model，也先做同样检查。

另外，`benchmark build` 现在也会用一个 **synthesis model** 来把模板和 slot 实现成自然语言场景。默认示例配置同样使用：

- `Qwen3-32B`

### 2.2 ethics sandbox 包存在且完整

默认路径：

- [package](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/package)

兼容 alias 仍然保留在：

- [sandbox_template](/Users/morinop/coding/whitzardgen/docs/ethics_design/sandbox_template)

至少应包含：

- `manifest.yaml`
- `slot_library.yaml`
- `analysis_codebook.yaml`
- `templates/*.yaml`

### 2.3 你知道当前 recipe 在做什么

默认 recipe：

- [ethics_structural.yaml](/Users/morinop/coding/whitzardgen/examples/experiments/ethics_structural.yaml)

它当前定义的是：

- benchmark builder：`ethics_sandbox`
- source package：`examples/benchmarks/ethics_sandbox/package`
- semantic synthesis model：`Qwen3-32B`
- targets：`Qwen3-32B`
- normalizer：`ethics_structural_normalizer`
- scorer：`ethics_structural_judge`
- analysis plugins：
  - `ethics_family_consistency`
  - `ethics_slot_sensitivity`

`benchmark build` 的产物本身就是可直接评测的 benchmark bundle，不需要再做任何导数步骤。最关键的文件是：

- `cases.jsonl`
  - 包含完整 case 文本以及 `slot_assignments`、`decision_frame`、`decision_options` 等 metadata
- `benchmark_manifest.json`
- `stats.json`
- `raw_realizations.jsonl`
  - build 阶段实际生成出来的 realization 结果，包含 spec、writer output 和 validation 状态
- `rejected_realizations.jsonl`
  - 没通过当前 validation gate 的 realization；它们不会进入 `cases.jsonl`，但也不会丢失
- `stats.json`

## 3. 推荐的运行方式

### 方式 A：直接跑 recipe

最推荐先用这个方式，因为它最贴近当前 benchmark-centric 架构：

```bash
whitzard evaluate run \
  --recipe examples/experiments/ethics_structural.yaml
```

这条命令会自动完成：

1. 用 ethics builder 采样 slot / 检查深层结构
2. 用 writer prompt + synthesis model 把结构化 spec 实现成第二人称自然场景
3. 为每个 case 同时生成结构化 A/B 选择支，默认只存储，不自动附加到 target 输入
4. 用 validator prompt 判断 benchmark-feel、冲突软化、第三路径、第二人称沉浸感和二选一结构是否合规，并在需要时重试
5. 构造 benchmark bundle
6. 把 benchmark bundle 编译成 `EvalTask`
7. 生成 `CompiledTaskPlan`
8. 通过 `RunEngineGateway` 调用现有 `run_flow` 执行 target model
9. 跑 ethics normalizer
10. 跑 scorer
11. 跑 group analysis 和 analysis plugins
12. 写出 V2 experiment bundle

### 方式 B：分两步跑

如果你想先检查 benchmark，再决定是否执行模型，建议用分步方式。

#### 第一步：构建 benchmark

```bash
whitzard benchmark build \
  --builder ethics_sandbox \
  --source examples/benchmarks/ethics_sandbox/package \
  --config examples/benchmarks/ethics_sandbox/example_build.yaml \
  --synthesis-model Qwen3-32B \
  --build-mode matrix
```

构建后先 inspect：

```bash
whitzard benchmark inspect runs/benchmarks/<benchmark_bundle_dir>
```

如果你只想关掉 build 阶段基于模型的 validator，而保留 deterministic guard，可以把 builder config 里的：

```yaml
validator:
  enabled: false
```

然后再 build。构建好的 bundle 仍然可以直接给 `whitzard evaluate run --benchmark ...` 使用。

当前行为也已经调整成：

- build 不会因为少量 invalid realization 直接失败
- 通过 validation 的 case 会进入 `cases.jsonl`
- 未通过的会进入 `rejected_realizations.jsonl`
- 所以 synthesis 已经生成出来的数据不会因为少数坏样本整批白跑

#### 第二步：执行 experiment

```bash
whitzard evaluate run \
  --benchmark runs/benchmarks/<benchmark_bundle_dir> \
  --targets Qwen3-32B \
  --normalizers ethics_structural_normalizer \
  --evaluators ethics_structural_judge \
  --analysis-plugins ethics_family_consistency ethics_slot_sensitivity
```

这里 CLI 参数名仍然保留 `--evaluators`，但在当前 V2 核心里，它会被解析成 **scorers**。

## 4. 当前架构下每一步实际在做什么

### 4.1 benchmark build

当前这一步已经不是“机械拼 prompt”，而是：

```text
slot sampling
-> deep-structure / invariant guard
-> realization template rendering
-> synthesis model via existing run kernel
-> final BenchmarkCase compilation
```

builder 最终会把 sandbox template 包转成 canonical `BenchmarkCase`，重点保留：

- `benchmark_id`
- `case_id`
- `input_modality`
- `input_payload`
- `metadata`
- `grouping`
- `execution_hints`
- `evaluation_hints`

对 ethics workload 来说，尤其重要的是这些 metadata 不会丢：

- `family_id` / `template_id`
- `variant_group_id`
- `slot_assignments`
- `deep_structure`
- `analysis_targets`
- `response_capture_contract`
- `decision_frame`
- `decision_options`
- `realization_prompt_template`
- `synthesis_model`
- `synthesis_request_version`
- `realization_provenance`

如果后续测试想自定义 evaluate 阶段真正发给 target model 的提示词，不需要改 benchmark build；直接在 experiment recipe 里配置 `execution_policy.target_prompt_template` 即可。推荐直接引用一个可编辑的模板文件，并用 `variable_allowlist` 显式声明模板可以访问哪些 case 字段和 metadata。

旧的 `execution_policy.text_prompt_composition.append_structured_choices` 仍然兼容，但现在更推荐使用 `target_prompt_template`。例如：

```yaml
execution_policy:
  target_prompt_template:
    path: templates/ethics_target_scene_with_choices.txt
    version: v1
    variable_allowlist:
      - prompt
      - formatted_choices
    helpers:
      - formatted_choices
    missing_variable_policy: warn_and_empty
```

judge/scorer 侧也支持同样的思路。你可以在 recipe 里放 `execution_policy.judge_prompt_template`，或者在 evaluator 配置里放 `prompt_template`，用来控制 judge 实际看到的 rubric prompt，同时继续保留原来的 annotation profile/output contract。

其中 `realization_prompt_template` 只标准记录“用了哪一个 build-time prompt template 名称”，core 不强制规定 profile schema。

### 4.2 task compile

当前 `evaluate run` 不再直接“拿 benchmark 然后临时调一堆 service”，而是先编译成：

- `EvalTask`
- `CompiledTaskPlan`
- `ExecutionRequest[]`

也就是说，真正被冻结并写盘的是：

- 用哪个 case set
- 跑哪些 targets
- 用哪些 normalizers
- 用哪些 scorers
- 用哪些 analysis plugins
- 最终有哪些 execution requests

### 4.3 target execution

执行层仍然复用你现有的 `run_flow` 内核。

这里只是通过：

- [gateway.py](/Users/morinop/coding/whitzardgen/src/whitzard/benchmarking/gateway.py)

把 `ExecutionRequest` 转成当前 run engine 还能接受的 prompt records。

所以：

- benchmark core 不直接侵入 run engine
- run engine 仍然保持 benchmark-agnostic
- 这也是当前架构里最重要的分层之一

### 4.4 normalization

normalizer 会把 target 输出整理成 `NormalizedResult`。

对 ethics 任务，通常会提取：

- `decision_text`
- `refusal_flag`
- `confidence_signal`
- `reasoning_trace_text`
- `extracted_fields`

这一步的作用是：

- 让后续 scorer / plugin 不必直接处理杂乱原始输出
- 让跨模型对比更稳定

### 4.5 scoring

V2 里主结果类型不再是 `EvaluatorResult`，而是：

- `ScoreRecord`

当前 scorer 有两类：

- rule scorer
- judge scorer

ethics 示例默认使用 judge scorer，因此会把 annotation/judge 结果统一回填成 `ScoreRecord`。

### 4.6 group analysis 和 analysis plugins

当前架构里这两层是分开的：

- core group analysis：做 generic 聚合
- analysis plugins：做领域特定分析

对 ethics 而言，examples 里现在已经接了两类 plugin：

- family consistency
- slot sensitivity

这正是为什么伦理逻辑应该留在 `examples/`，而不是写进 core。

## 5. 输出结果怎么看

experiment 完成后，会写出一个 V2-first bundle。当前最重要的文件是：

- `cases.jsonl`
- `execution_requests.jsonl`
- `target_results.jsonl`
- `normalized_results.jsonl`
- `score_records.jsonl`
- `group_analysis_records.jsonl`
- `analysis_plugin_results.jsonl`
- `experiment_log.jsonl`
- `compiled_task_plan.json`
- `experiment_manifest.json`
- `summary.json`
- `report.md`
- `failures.json`

建议这样理解：

- `cases.jsonl`
  - benchmark case 本体，保留 structural lineage
- `execution_requests.jsonl`
  - 真正发给 target plane 的执行请求
- `target_results.jsonl`
  - target model 执行后的直接结果映射
- `normalized_results.jsonl`
  - ethics-aware 规范化中间层
- `score_records.jsonl`
  - scorer/judge 输出的统一记录
- `group_analysis_records.jsonl`
  - generic group aggregation 结果
- `analysis_plugin_results.jsonl`
  - ethics-specific comparative analysis 结果
- `experiment_log.jsonl`
  - evaluation-plane 的 append-only 事件日志
- `compiled_task_plan.json`
  - 冻结后的实验计划，适合追溯和复现

查看方式：

```bash
whitzard evaluate inspect <experiment_id>
whitzard experiments report <experiment_id>
```

## 6. 最常见的修改点

### 6.1 想换 target models

直接改 recipe 的 `targets`，或者命令行覆盖：

```bash
whitzard evaluate run \
  --recipe examples/experiments/ethics_structural.yaml \
  --targets Qwen3-32B Another-LLM
```

### 6.2 想调 benchmark 规模

修改：

- [example_build.yaml](/Users/morinop/coding/whitzardgen/examples/benchmarks/ethics_sandbox/example_build.yaml)

例如：

```yaml
realizations_per_template: 8
```

### 6.3 想改伦理结构本身

改这里：

- [sandbox_template](/Users/morinop/coding/whitzardgen/docs/ethics_design/sandbox_template)

重点通常是：

- `templates/*.yaml`
- `slot_library.yaml`
- `analysis_codebook.yaml`

### 6.4 想换 scorer / judge model

你可以：

- 直接换 `evaluators` 配置
- 或继续用 legacy judge flags：

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

在 V2 内部，这也会被收敛到 scorer 语义。

## 7. 推荐的实际操作顺序

### 冒烟测试

先把规模压到很小，比如每个模板 `1` 到 `2` 个 realization，然后：

```bash
whitzard evaluate run --recipe examples/experiments/ethics_structural.yaml
```

重点看：

- benchmark 是否构建成功
- `compiled_task_plan.json` 是否符合预期
- `normalized_results.jsonl` 是否抽出了 decision / refusal / reasoning
- `score_records.jsonl` 是否有稳定字段
- `analysis_plugin_results.jsonl` 是否真的按 family / slot 输出了聚合结果

### 中规模实验

确认冒烟没问题后，再提高：

- `realizations_per_template`
- target model 数量
- scorer / plugin 组合

更建议先逐步加规模，而不是一开始就做很重的多模型 sweep。

## 8. 当前你需要记住的一条核心原则

在当前架构下运行伦理冲突评测时，**最重要的不是 prompt 文案本身，而是 structural lineage 不丢失**。

也就是这些东西必须一路保留下去：

- template / family identity
- sibling / variant grouping
- slot assignments
- response capture contract
- scorer 与 analysis plugin 的产物 lineage

因为伦理冲突评测真正要分析的，不只是“模型回答了什么”，而是：

- 同一结构下是否稳定
- 不同 slot 扰动下是否翻转
- 家族内部是否一致
- 哪些 norm / value / tradeoff 被优先化

如果这些 lineage 丢了，后面的规模化研究就失去意义了。
