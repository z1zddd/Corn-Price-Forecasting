# Registry

这里负责模型注册和工厂分发。

`factory.py` 根据配置里的 `type`、`name` 和 `params` 创建对应模型。新增模型时优先在具体模型目录实现，再在这里接入。

