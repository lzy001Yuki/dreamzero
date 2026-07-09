# Find9Shape Annotation Context Training

这份文档记录新的 find9shape 训练/测试流程。新逻辑是：每个 episode 从标注图路径中解析出上下文边界 `i`，把 `0..i` 的 observation 作为 DreamZero 的上下文，然后从第 `i` 帧开始监督模型预测后面 `N` 帧。如果后续帧数不够，则用最后一帧/最后一个 action padding。

## 相关文件

- 训练脚本：`scripts/train/find9shape_annotation_context_training.sh`
- 数据配置：`groot/vla/configs/data/dreamzero/find9shape_annotation_context.yaml`
- Dataset 实现：`groot/vla/data/dataset/lerobot_sharded.py`
  - `ShardedLeRobotAnnotationContextDataset`
  - `ShardedLeRobotAnnotationContextMixtureDataset`
- 查看 episode 到 `i` 的映射：`scripts/data/print_find9shape_annotation_indices.py`
- 测试 client：`eval_utils/infer_find9shape_small_client.py`
- 推理 server：`eval_utils/serve_find9shape_small.py`

## `i` 的来源

`i` 不是直接读取 `annotation.csv` 的 `frame_index` 列，而是从 `top_marked_frame_path` 里解析：

```text
/home/yan/myProjects/GenMem/annotation/shape_prompt_annotations/marked_frames/top/episode_000000_frame_000003_red_point.png
```

上面的路径会解析出：

```text
episode_000000: i=3
```

解析规则匹配文件名里的这一段：

```text
episode_000000_frame_000003_red_point.png
```

即取 `frame_000003` 中的数字作为 `i`。

## 训练数据窗口

对每个 episode：

- 从 `annotation.csv` 的 `top_marked_frame_path` 解析出 `i`
- 视频 observation 从 episode 开头取，即从第 `0` 帧开始
- state/action 从第 `i` 帧开始取
- action horizon 默认为 `8`
- `num_frames` 默认为 `17`
- 如果从 `i` 往后不足 `action_horizon`，action 用最后一个 action padding
- 如果视频索引超过 episode 末尾，视频 loader 会 clamp 到最后一帧，相当于最后一帧 padding

当前训练脚本里的关键设置：

```bash
num_frames=17
action_horizon=8
num_views=2
num_frame_per_block=4
num_action_per_block=8
num_state_per_block=1
```

## 检查 Annotation 映射

先确认 `annotation.csv` 存在且包含 `top_marked_frame_path`：

```bash
ls -l annotation.csv
```

输出每个 episode 对应的 `i`：

```bash
python3 scripts/data/print_find9shape_annotation_indices.py \
  --annotation_csv annotation.csv
```

示例输出：

```text
episode_000000: i=3
episode_000001: i=6
episode_000002: i=6
...
```

如果这里报错，优先检查：

- `annotation.csv` 是否为空
- 是否有 `episode_index` 列
- 是否有 `top_marked_frame_path` 列
- `top_marked_frame_path` 是否形如 `episode_000000_frame_000003_red_point.png`

## 运行训练

最简单启动方式：

```bash
bash scripts/train/find9shape_annotation_context_training.sh
```

常用覆盖项：

```bash
FIND9SHAPE_DATA_ROOT=/path/to/find9shape_small \
FIND9SHAPE_ANNOTATION_CSV=/path/to/annotation.csv \
OUTPUT_DIR=/path/to/output_dir \
MAX_STEPS=1000 \
SAVE_STEPS=500 \
NUM_GPUS=1 \
bash scripts/train/find9shape_annotation_context_training.sh
```

脚本默认路径：

```bash
FIND9SHAPE_DATA_ROOT=$REMOTE_DATASET_ROOT/find9shape_small
FIND9SHAPE_ANNOTATION_CSV=$PWD/annotation.csv
OUTPUT_DIR=$REMOTE_CHECKPOINT_ROOT/dreamzero_find9shape_annotation_context_lora
```

训练脚本会使用新的 data config：

```bash
data=dreamzero/find9shape_annotation_context
```

这个 config 会切到 annotation-context dataset：

```yaml
mixture_dataset_cls: groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotAnnotationContextMixtureDataset.from_mixture_spec
single_dataset_cls: groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotAnnotationContextDataset
```

训练初始化时会打印从 `top_marked_frame_path` 解析出来的 episode 到 `i` 的映射。

## 测试流程

### 1. 启动 Server

如果要让测试时第一步真的吃到 `[0, i]` 的上下文，需要给 server 加 `--first_call_full_video`：

```bash
python eval_utils/serve_find9shape_small.py \
  --model_path /path/to/checkpoint \
  --port 8010 \
  --first_call_full_video
```

如果不加 `--first_call_full_video`，当前 causal server 在第一次请求时默认只使用 1 帧，这样就不会真正用到完整上下文。

### 2. 运行 Client

从 `annotation.csv` 自动解析 episode 的 `i` 并发送 `[0, i]` 上下文：

```bash
python eval_utils/infer_find9shape_small_client.py \
  --dataset_root ./datasets/find9shape_small \
  --annotation_csv ./annotation.csv \
  --episode_index 0 \
  --use_annotation_start \
  --chunk_size 8 \
  --port 8010
```

client 会打印：

```text
Using annotation start: episode_000000: i=3
First request will send frames [0, i] as context.
```

如果想持续发送从 episode 开头到当前帧的 history，可以加：

```bash
--send_history
```

完整示例：

```bash
python eval_utils/infer_find9shape_small_client.py \
  --dataset_root ./datasets/find9shape_small \
  --annotation_csv ./annotation.csv \
  --episode_index 0 \
  --use_annotation_start \
  --send_history \
  --chunk_size 8 \
  --port 8010
```

## 推荐完整流程

1. 检查 annotation 文件：

```bash
ls -l annotation.csv
```

2. 打印并确认 `episode -> i`：

```bash
python3 scripts/data/print_find9shape_annotation_indices.py \
  --annotation_csv annotation.csv
```

3. 启动训练：

```bash
FIND9SHAPE_DATA_ROOT=/path/to/find9shape_small \
FIND9SHAPE_ANNOTATION_CSV=/path/to/annotation.csv \
OUTPUT_DIR=/path/to/output_dir \
MAX_STEPS=1000 \
bash scripts/train/find9shape_annotation_context_training.sh
```

4. 训练结束后启动 server：

```bash
python eval_utils/serve_find9shape_small.py \
  --model_path /path/to/output_dir/checkpoint-XXXX \
  --port 8010 \
  --first_call_full_video
```

5. 运行测试 client：

```bash
python eval_utils/infer_find9shape_small_client.py \
  --dataset_root /path/to/find9shape_small \
  --annotation_csv /path/to/annotation.csv \
  --episode_index 0 \
  --use_annotation_start \
  --chunk_size 8 \
  --port 8010
```

## 注意事项

- `annotation.csv` 不能是空文件。
- `top_marked_frame_path` 必须包含 `episode_XXXXXX_frame_YYYYYY_red_point.png` 这种文件名。
- 新训练脚本不会使用 CSV 的 `frame_index` 列来决定 `i`。
- 如果一个 episode 在 CSV 里出现多次，且路径解析出来的 `i` 不一致，会直接报错。
- 当前逻辑每个 episode 只用 annotation 对应的一个 `i` 作为训练起点。
- 旧的 `scripts/train/find9shape_training.sh` 不受影响。
