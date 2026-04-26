# VehicleMemBench

多用户长期记忆的车载代理基准测试。
相关论文：<https://arxiv.org/abs/2603.23840>

## 运行环境

- Python 3.12+
- `uv` 包管理器

## 实验组

### A. 模型评估

评估 backbone 模型在不同记忆构建方式下的表现：

- **Raw History（`none`）**：不提供任何历史信息，测试模型零样本推理能力
- **Gold Memory（`gold`）**：提供 ground-truth 最新用户偏好，代表理论性能上界
- **Recursive Summarization（`summary`）**：将历史压缩为分层摘要，测试对蒸馏信息的推理
- **Key-Value Store（`key_value`）**：将偏好组织为结构化属性-值对，测试精确索引检索

### B. 内存系统评估

先摄入对话历史到外部记忆系统，再基于检索到的记忆评估 agent 能否执行正确车辆操作。支持的系统：

- **内建系统**：MemoryBank（基于 FAISS 的本地向量检索）
- **第三方系统**：Mem0 / MemOS / LightMem / Supermemory / Memobase

### C. MemoryBank

基于向量检索的本地记忆系统，使用 OpenAI Embedding API（或兼容接口）将历史编码为向量，通过 FAISS 进行相似性检索，支持反思性摘要（reflection）机制。

## 数据集

- **50 个可执行样本**，覆盖多用户场景（主驾/副驾偏好冲突、跨会话记忆等）
- **`benchmark/qa_data/qa_{1..50}.json`**：每个样本包含用户查询、ground-truth 工具调用序列和期望的最终车辆状态
- **`benchmark/history/history_{1..50}.txt`**：对应的多轮对话历史，模拟长期人车交互
- **23 个车辆模块**：座椅、窗户、空调、导航、音乐、灯光、雨刮、门、天窗、遮阳帘、后视镜、方向盘、HUD、仪表盘、中控屏、蓝牙、视频、收音机、踏板、油箱口、前备箱、后备箱、头顶屏幕等

## 实验过程

1. **输入**：用户查询 + 对话历史（或经由记忆系统检索的上下文）
2. **推理**：LLM 调用 vehicle API 生成 Python 工具调用代码
3. **执行**：模拟器执行工具调用，更新车辆环境状态
4. **评估**：比较最终状态与 ground-truth，计算指标
5. **输出**：结果写入 `log/`（模型评估）或 `memory_system_log/`（系统评估）

## 评测指标

- **Exact State Match**：最终环境状态是否与 ground-truth 完全一致（严格二值指标）
- **Field-level Precision/Recall/F1**：字段维度的精确率、召回率、F1（正确变更 vs 错误变更）
- **Value-level Precision/Recall/F1**：值维度的精确率、召回率、F1（变更值是否正确）
- **Tool Call Count**：平均每次任务的工具调用次数，衡量执行开销

## 项目结构

- `benchmark/qa_data/` — 50 个 JSON 格式的可执行 QA 样本
- `benchmark/history/` — 50 个对应的对话历史记录
- `environment/` — 包含 23 个车辆模块的模拟器（座椅、窗户、空调、导航等）
- `evaluation/` — 模型和内存系统评估框架
- `scripts/` — 测试运行脚本

## 代码风格

- 实现基于带类型注解的 Python
- 使用 `uv` 进行包管理（`uv.lock` 已锁定）
- 导入顺序：标准库 → 第三方库 → 本地模块
- 使用 `uv run` 执行脚本（`model_test.sh` / `memorysystem_test.sh` 仍使用 `python` / `py`）
- 输出写入 `log/`，在 `.gitignore` 中排除
- 评估指标：精确状态匹配、字段/值级别的精确率/召回率/F1、工具调用次数
