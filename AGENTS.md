# VehicleMemBench

多用户长期记忆的车载代理基准测试。
相关论文：https://arxiv.org/abs/2603.23840

## 启动命令

- 设置环境变量：`export LLM_API_BASE="..." LLM_API_KEY="..." LLM_MODEL="..."`
- 模型评估：`bash scripts/model_test.sh`
- 内存系统评估：`bash scripts/memorysystem_test.sh`
- 交互式运行：`bash scripts/base_test.sh`
- 单个 Python 脚本：`uv run evaluation/model_evaluation.py ...`

## 项目结构

- `benchmark/qa_data/` — 50 个 JSON 格式的可执行 QA 样本
- `benchmark/history/` — 50 个对应的对话历史记录
- `environment/` — 包含 23 个车辆模块的模拟器（座椅、窗户、空调、导航等）
- `evaluation/` — 模型和内存系统评估框架
- `scripts/` — 测试运行脚本

## 代码风格

- 作者使用带类型注解的 Python
- 使用 `uv` 进行包管理（`uv.lock` 已锁定）
- 导入顺序：标准库 → 第三方库 → 本地模块
- 使用 `uv run` 执行脚本，而非 `python` 或 `pip`
- 输出写入 `log/`，在 `.gitignore` 中排除
- 评估指标：精确状态匹配、字段/值级别的精确率/召回率/F1、工具调用次数
