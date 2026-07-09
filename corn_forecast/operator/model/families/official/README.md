# Official Model Pool

这里定义官方 57 模型池。

- `base.py`: 57 个模型名称、家族分组和 `OfficialPoolSpec` 数据结构。
- `pool.py`: 命名模型池展开和规格组装。
- `adapter.py`: 把官方池模型包装成框架统一接口。
- `io.py`: 输入形状、概率输出和数值转换工具。
- `tabular/`: 传统表格模型规格。
- `aeon/`: aeon 时间序列模型规格。
- `keras/`: Keras 序列模型规格。
