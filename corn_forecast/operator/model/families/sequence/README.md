# Sequence Models

这里放面向时间序列窗口输入的深度序列模型。每个模型只定义模型本身及其
项目接口；滚动切分、训练编排、评估和报告仍在 `corn_forecast.pipeline`。

每个复杂方法单独放一个脚本，例如 `lstm.py`、`gru.py`、`transformer.py`、
`patchtst.py`、`itransformer.py`、`dlinear.py` 和 `dual_stream_lstm.py`。

## Upstream-source adapters

`simpletm.py`、`timemixer.py`、`tide.py` 和 `xlinear.py` 是明确的模型方法
适配器。它们不复制或修改上游源码，而是通过 `params.source_root` 加载用户
本地的上游 checkout：

| model | upstream | required module |
| --- | --- | --- |
| `simpletm` | `https://github.com/vsingh-group/SimpleTM` | `model/SimpleTM.py` |
| `timemixer` | `https://github.com/kwuking/TimeMixer` | `models/TimeMixer.py` |
| `tide` | `https://github.com/thuml/Time-Series-Library` | `models/TiDE.py` |
| `xlinear` | `https://github.com/Zaiwen/XLinear` | `models/XLinear.py` |

适配器的输入仍是本项目的 `[sample, feature, lookback]` 窗口。它只使用每个
rolling fold 的训练集拟合连续 `target_return_fwd`，把预测收益的正负转换为
方向；不会使用上游脚本中的固定 70/10/20 切分。

本项目已经在每个 fold 内做了 train-only 序列标准化，因此 XLinear、SimpleTM
和 TimeMixer 默认关闭其上游的内部归一化。TiDE 的上游实现内置价格尺度的反
归一化，适配器在其输出后接一个可训练的收益头；这使训练目标保持为
`target_return_fwd`，而不是把价格预测误当作收益预测。

最小 YAML 形态如下。`source_root` 必须是本机实际 checkout 的绝对路径，不能
写入仓库配置。

```yaml
models:
  - name: xlinear
    type: xlinear
    params:
      source_root: /absolute/path/to/XLinear
      d_model: 64
      epochs: 20
      batch_size: 8
      lr: 0.001
```

上游源码是可选依赖；没有 `source_root` 时，创建对应模型会给出明确错误，但
不会影响其他模型和 CLI 的 import。
