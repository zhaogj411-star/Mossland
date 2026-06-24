# Eval Benchmark 接口说明

本页用于快速恢复 `scripts/mossland-codec/eval_benchmark/` 的人工接模型和启动评测流程。

## 目录职责

- `manifest.py`：定义评测样本 `EvalItem`，支持 JSONL/JSON/CSV manifest。
- `audio_io.py`：统一音频读取、重采样、保存和 reference path 选择。
- `model_adapters.py`：统一模型推理接口。新增普通模型优先接这里。
- `run.py`：单进程推理和指标入口；可直接跑 Mossland checkpoint、自定义 adapter，或只评已有 `prediction_path`。
- `run_sharded_metric_eval.py`：GPU 指标/推理分片入口，最后自动合并 summary。
- `run_sharded_visqol_eval.py`：CPU-only ViSQOL 分片入口。
- `aggregate_results.py`：合并多个 `results.jsonl`，重新聚合距离指标和 FAD。
- `baselines/`：历史官方 baseline adapter。只有当第三方模型需要复杂依赖、下载、CLI 调度或任务筛选时，才新增独立文件。
- `build_*manifest.py`、`download_*`：数据和 manifest 准备工具。
- `generate_*table*.py`、`tables/`：已有结果表格生成，不负责推理。

## 最小模型接口

普通模型只需要暴露 `module:object`，其中 object 可以是 adapter 实例，也可以是返回 adapter 的 factory。adapter 必须实现：

```python
def predict(item, source, target, context):
    return prediction
```

- `item`：`EvalItem`，含 `task_id`、路径、采样率、seed 和 metadata。
- `source`：已按任务处理好的输入音频，shape `[C, T]`。
- `target`：reference 音频；有些任务或自回归推理可忽略。
- `context.device`：命令行 `--device`。
- `context.options`：命令行 `--adapter-config` 传入的 JSON dict。
- 返回值：`torch.Tensor` 或可转 tensor 的数组，shape 必须是 `[C, T]` 或 `[T]`；`run.py` 会保存成 wav，并继续算指标。

最小例子：

```python
# tmp/my_eval_adapter.py
import torch


class MyAdapter:
    def __init__(self, context):
        self.device = context.device or "cuda"

    @torch.inference_mode()
    def predict(self, item, source, target, context):
        # source 是 [C,T]，这里替换成真实模型推理。
        return source


def create_adapter(context):
    return MyAdapter(context)
```

启动 1 条 smoke：

```sh
PYTHONPATH=$PWD \
python -m scripts.mossland-codec.eval_benchmark.run \
  --manifest scripts/mossland-codec/eval_benchmark/data/example_manifest.jsonl \
  --adapter-target tmp.my_eval_adapter:create_adapter \
  --adapter-config '{"checkpoint":"ckpt/my_model.pt"}' \
  --device cuda:0 \
  --metrics-device cuda:0 \
  --fad-backend none \
  --max-items 1 \
  --progress-every 1 \
  --output-dir tmp/eval_smoke_my_model
```

确认 `tmp/eval_smoke_my_model/predictions/` 有 wav、`summary.json` 能写出后，再把 `--max-items 1` 去掉或改成分片运行。

## Mossland checkpoint

旧用法仍然保留，不需要写 adapter：

```sh
PYTHONPATH=$PWD \
python -m scripts.mossland-codec.eval_benchmark.run \
  --manifest scripts/mossland-codec/eval_benchmark/data/example_manifest.jsonl \
  --checkpoint-dir ckpt/mosslandcodec0617_rvq64 \
  --device cuda:0 \
  --metrics-device cuda:0 \
  --fad-backend clap \
  --output-dir scripts/mossland-codec/eval_benchmark/runs/mosslandcodec0617_rvq64_eval
```

`--quantized` 会传给 Mossland checkpoint adapter；自定义 adapter 也能在 `context.options["quantized"]` 读到该值。

## 已有 prediction manifest

如果已经有 wav，只需在 manifest 中写 `prediction_path`，不要传 `--checkpoint-dir` 或 `--adapter-target`：

```sh
PYTHONPATH=$PWD \
python -m scripts.mossland-codec.eval_benchmark.run \
  --manifest tmp/my_model_prediction_manifest.jsonl \
  --metrics-device cuda:0 \
  --fad-backend clap \
  --output-dir tmp/my_model_eval_clap
```

## 正式分片评测

GPU 指标或可并行推理使用：

```sh
PYTHONPATH=$PWD \
python -m scripts.mossland-codec.eval_benchmark.run_sharded_metric_eval \
  --manifest scripts/mossland-codec/eval_benchmark/data/example_manifest.jsonl \
  --adapter-target tmp.my_eval_adapter:create_adapter \
  --adapter-config '{"checkpoint":"ckpt/my_model.pt"}' \
  --output-dir tmp/my_model_eval_clap \
  --shard-output-dir tmp/my_model_eval_clap_shards \
  --fad-backend clap \
  --gpus 0,1,2,3 \
  --num-shards 4 \
  --progress-every 100
```

如果已经提前生成 prediction manifest，则删掉 `--adapter-target/--adapter-config`，把 `--manifest` 换成带 `prediction_path` 的 JSONL。

ViSQOL 是 CPU-only，用 `run_sharded_visqol_eval.py`，不要占 GPU。

## 什么时候新增 `baselines/*.py`

优先用 `--adapter-target`。只有出现以下情况才新增独立 baseline 文件：

- 需要下载或校验官方 checkpoint。
- 需要调用第三方 CLI 或 repo。
- 一个模型需要批量处理多任务组，例如 source separation 一次输出多个 stem。
- 需要复现论文/官方 baseline，并保留固定命令供后续表格复现。

新增 baseline 时复用 `baselines/common.py` 的 `add_common_args()`、`baseline_prediction_path()`、`write_predicted_manifest()`，最后输出带 `prediction_path` 的 JSONL，再交给 `run.py` 算统一指标。
