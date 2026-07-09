# Model Scripts

这里是玉米预测脚本入口目录。

脚本用于运行、同步、聚合、数据准备和临时维护，不属于 `operator/model` 模型算子本体。脚本之间仍保持同目录引用，远程运行脚本默认使用：

```bash
${ROOT_DIR}/玉米预测/scripts
```

如果远程机器还没同步新目录结构，可以通过 `MODEL_SCRIPT_DIR` 或 `REMOTE_SCRIPT_DIR` 覆盖脚本路径。

