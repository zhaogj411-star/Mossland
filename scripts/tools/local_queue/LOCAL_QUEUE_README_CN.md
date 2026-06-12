# Local Queue 中文说明

这是一套基于共享持久化目录的本地任务队列框架，用来在低优先级、可抢占、机器数量不固定的环境里做数据处理。

它不依赖 Postgres、Redis 或外部网络服务。所有机器只要能访问同一个共享目录，就可以用同一个命令启动 worker。

## 适用场景

- 平台会动态分配不同数量的机器或 GPU。
- 机器只能挂低优先级，随时可能被 kill。
- 部分机器不能联网。
- 所有机器能访问同一个持久化目录。
- 有多种数据处理任务，希望复用同一套任务领取、去重、重试、监控流程。

核心原则：

```text
任务可以重复执行，但结果只能成功提交一次。
worker 可以随时死亡，但任务租约必须会过期。
业务代码只写 tmp 输出，最终提交由框架负责。
```

## 目录结构

假设共享根目录是 `/shared/data_pipeline`，一个 job 叫 `dataset_v1.pipeline_v3`，目录会长这样：

```text
/shared/data_pipeline/
  manifests/
    dataset_v1.pipeline_v3/
      manifest.json
      buckets/
        00.jsonl
        01.jsonl
        ...

  leases/
    dataset_v1.pipeline_v3/
      ab/
        <task_id>.lock/
          lease.json

  state/
    dataset_v1.pipeline_v3/
      done/
        ab/
          <task_id>.json
      failed/
        ab/
          <task_id>.json

  outputs/
    dataset_v1.pipeline_v3/
      ab/
        <handler-output-file>
        <handler-output-file>.meta.json

  tmp/
    <worker_id>/

  heartbeats/
    dataset_v1.pipeline_v3/
      <worker_id>.json

  logs/
    dataset_v1.pipeline_v3/
      errors/
        ab/
          <task_id>.<time>.<worker>.json

  final/
    dataset_v1.pipeline_v3/
      summary.json
      _SUCCESS
```

## 关键概念

### Job

一个 job 表示一次完整的数据处理运行，例如：

```text
dataset_v1.pipeline_v3
audio_v2.decode_v1
text_v5.clean_v4
```

建议 job 名包含：

- 数据版本
- 处理逻辑版本
- 配置版本或 hash

不要把不同处理逻辑混在同一个 job 里。

### TaskHandler

每种业务任务实现一个 `TaskHandler`。

框架负责：

- 写 manifest
- 抢任务
- 创建 lease
- 刷新心跳
- 任务重试
- 失败记录
- 原子提交输出
- 写 done marker
- 监控进度
- finalize 检查

业务 handler 只负责：

- 如何生成任务
- 如何处理单个任务

### TaskRecord

一个 `TaskRecord` 是一个稳定的任务分片。

它包含：

```python
TaskRecord(
    task_id="stable-task-id",
    task_type="my_task",
    payload={...},
    priority=0,
)
```

`task_id` 必须稳定。推荐用输入路径、分片范围、处理版本、配置 hash 算出来：

```python
task_id = sha256(input_path + start + end + job + config_hash)
```

这样同一个任务无论在哪台机器上跑，ID 都一样。

### TaskResult

handler 处理完成后返回：

```python
TaskResult(
    output_path=tmp_output,
    metadata={"rows": 123},
)
```

注意：`output_path` 必须是写在 `tmp_dir` 下面的临时文件。不要直接写最终 `outputs` 目录。

## 如何接入一种新任务

新建一个 handler 文件，例如 `my_handlers/audio_decode.py`：

```python
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib

from local_queue import TaskHandler, TaskRecord, TaskResult


class AudioDecodeHandler(TaskHandler):
    task_type = "audio_decode"

    def build_tasks(
        self,
        root: Path,
        job: str,
        config: Mapping[str, Any],
    ) -> Iterable[TaskRecord]:
        inputs = config["inputs"]
        config_hash = config.get("config_hash", "dev")

        for audio_path in inputs:
            raw = f"{audio_path}|{job}|{config_hash}"
            task_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()

            yield TaskRecord(
                task_id=task_id,
                task_type=self.task_type,
                payload={"input": audio_path},
            )

    def process(
        self,
        task: TaskRecord,
        *,
        root: Path,
        job: str,
        worker_id: str,
        device: str,
        tmp_dir: Path,
    ) -> TaskResult:
        input_path = Path(task.payload["input"])
        output = tmp_dir / f"{task.task_id}.jsonl"

        # TODO: 这里写真实业务逻辑
        # 只允许写 tmp_dir，不要直接写 outputs。
        with output.open("w", encoding="utf-8") as f:
            f.write(f'{{"input":"{input_path}","device":"{device}"}}\n')

        return TaskResult(
            output_path=output,
            metadata={"input": str(input_path)},
        )
```

handler 加载格式是：

```text
模块路径:类名
```

例如：

```text
my_handlers.audio_decode:AudioDecodeHandler
```

## 初始化任务

先准备配置文件，例如 `config.json`：

```json
{
  "inputs": [
    "/shared/raw/a.wav",
    "/shared/raw/b.wav"
  ],
  "config_hash": "decode_config_v1"
}
```

然后生成 manifest：

```bash
python -m local_queue.cli init \
  --root /shared/data_pipeline \
  --job audio_v1.decode_v1 \
  --handler my_handlers.audio_decode:AudioDecodeHandler \
  --config config.json
```

这会生成：

```text
manifests/audio_v1.decode_v1/manifest.json
manifests/audio_v1.decode_v1/buckets/*.jsonl
```

manifest 一旦生成，建议不要就地修改。如果处理逻辑或配置变了，新建一个 job。

## 启动 worker

所有机器都可以运行同一个命令：

```bash
python -m local_queue.cli start \
  --root /shared/data_pipeline \
  --job audio_v1.decode_v1 \
  --handler my_handlers.audio_decode:AudioDecodeHandler
```

默认会自动检测 GPU：

- 有 GPU：每张 GPU 启动一个 worker。
- 没有 GPU：启动一个 CPU worker。

也可以手动指定设备：

```bash
python -m local_queue.cli start \
  --root /shared/data_pipeline \
  --job audio_v1.decode_v1 \
  --handler my_handlers.audio_decode:AudioDecodeHandler \
  --devices cuda:0,cuda:1
```

如果只想启动一个 worker：

```bash
python -m local_queue.cli worker \
  --root /shared/data_pipeline \
  --job audio_v1.decode_v1 \
  --handler my_handlers.audio_decode:AudioDecodeHandler \
  --device cpu
```

如果只想跑一个任务就退出，用于测试：

```bash
python -m local_queue.cli worker \
  --root /shared/data_pipeline \
  --job audio_v1.decode_v1 \
  --handler my_handlers.audio_decode:AudioDecodeHandler \
  --device cpu \
  --once
```

## 监控进度

```bash
python -m local_queue.cli monitor \
  --root /shared/data_pipeline \
  --job audio_v1.decode_v1
```

输出类似：

```text
total: 100000
done: 82310
failed: 12
running_or_leased: 128
workers_seen: 64
remaining: 17678
```

字段含义：

- `total`：manifest 里的任务总数。
- `done`：已经成功完成的任务数。
- `failed`：达到最大失败次数后标记失败的任务数。
- `running_or_leased`：当前有 lease 的任务数，不一定都还活着，过期后会被其他 worker 重抢。
- `workers_seen`：写过 heartbeat 的 worker 数。
- `remaining`：还没 done 或 failed 的任务数。

## Finalize

所有任务跑完后执行：

```bash
python -m local_queue.cli finalize \
  --root /shared/data_pipeline \
  --job audio_v1.decode_v1
```

它会写：

```text
final/audio_v1.decode_v1/summary.json
```

只有在所有任务都有 done marker，并且没有 failed 任务时，才会写：

```text
final/audio_v1.decode_v1/_SUCCESS
```

如果允许存在 failed 任务也生成 `_SUCCESS`：

```bash
python -m local_queue.cli finalize \
  --root /shared/data_pipeline \
  --job audio_v1.decode_v1 \
  --allow-failed
```

## 去重和一致性

这个框架的去重依赖三件事：

1. `task_id` 稳定。
2. 输出先写 `tmp`，再由框架移动到 `outputs`。
3. 最终结果只信任 `state/<job>/done`，不要直接扫 `outputs`。

worker 被 kill 时可能留下：

```text
tmp/<worker_id>/...
leases/<job>/<prefix>/<task_id>.lock/lease.json
```

这没关系：

- tmp 文件不会被 finalize 信任。
- lease 超过 TTL 后会被其他 worker 回收。
- 同一个 task 即使被重复执行，也只有一个 done marker 被最终认可。

## Lease 参数

默认参数：

```text
lease_ttl = 300 秒
heartbeat_interval = 30 秒
max_attempts = 5
```

可以启动时调整：

```bash
python -m local_queue.cli start \
  --root /shared/data_pipeline \
  --job audio_v1.decode_v1 \
  --handler my_handlers.audio_decode:AudioDecodeHandler \
  --lease-ttl 600 \
  --heartbeat-interval 60 \
  --max-attempts 3
```

建议：

- 单个 task 处理时间控制在 1 到 10 分钟。
- `lease_ttl` 要明显大于 `heartbeat_interval`。
- 如果任务很长，handler 内部最好定期落自己的业务中间状态。

## 多种任务如何组织

推荐一个 job 对应一种 handler：

```text
audio_raw_v1.decode_v1       -> AudioDecodeHandler
audio_decode_v1.qc_v2        -> AudioQcHandler
text_raw_v3.clean_v4         -> TextCleanHandler
text_clean_v4.dedup_v1       -> TextDedupHandler
```

这样每个 job 的 manifest、done、failed、outputs 都是独立的，恢复和排查都更简单。

不建议把完全不同的业务任务混进同一个 job。虽然框架里有 `task_type` 字段，但第一版最稳的用法是：

```text
一个 job + 一个 handler + 一套输出目录
```

## 共享文件系统要求

共享目录最好支持：

```text
mkdir 原子
同一文件系统内 rename / os.replace 原子
文件修改能被其他机器及时看见
目录 list 不会长时间延迟
```

通常可用：

```text
NFS
Lustre
CephFS
BeeGFS
```

需要小心：

```text
S3Fuse
OSSFuse
对象存储挂载
某些云盘挂载
```

对象存储挂载的 `rename` 和 `list` 语义可能不可靠，必须先做压测。

## 本地示例

仓库里有一个示例 handler：

[examples/copy_jsonl_handler.py](/C:/Users/44184/Documents/moss/examples/copy_jsonl_handler.py)

它会把 JSONL 输入按行数切成多个任务，然后每个任务复制一段行到输出。

配置示例：

```json
{
  "inputs": [
    "C:/data/input.jsonl"
  ],
  "shard_lines": 10000,
  "config_hash": "copy_v1"
}
```

初始化：

```bash
python -m local_queue.cli init \
  --root C:/shared/data_pipeline \
  --job copy_demo_v1 \
  --handler examples.copy_jsonl_handler:CopyJsonlHandler \
  --config config.json
```

运行一个 worker：

```bash
python -m local_queue.cli worker \
  --root C:/shared/data_pipeline \
  --job copy_demo_v1 \
  --handler examples.copy_jsonl_handler:CopyJsonlHandler \
  --device cpu
```

监控：

```bash
python -m local_queue.cli monitor \
  --root C:/shared/data_pipeline \
  --job copy_demo_v1
```

## 文件清理建议

可以定期清理：

```text
tmp/<worker_id>/
leases/<job>/**/*.expired.*
heartbeats/<job>/*.json
logs/<job>/errors/
```

但不要在 job 运行中清理：

```text
manifests/
state/done/
state/failed/
outputs/
```

这些是恢复和 finalize 的依据。

## 生产使用建议

- 所有任务输出都由 `task_id` 决定，避免随机文件名污染最终结果。
- handler 的 `process()` 尽量做到确定性：同样输入和配置得到同样输出。
- 如果业务里有随机数，seed 用 `task_id` 派生。
- 每个 task 不要太大，避免低优先级被 kill 后浪费太多。
- 每个 task 也不要太小，否则共享文件系统压力会很高。
- 大规模任务建议按百万级以下一个 job 分批跑，便于排查和重跑。
- 最终下游只读取 `final/<job>/summary.json` 里的 outputs，或者只读取 done marker 对应的输出。
