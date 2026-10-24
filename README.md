# 云端语义风险辅助的微型无人机自主导航

本项目将云端 Qwen2.5-VL / LoRA 风险感知与本地 GRU-IQN / ART 导航连接起来。云端异步提供语义风险，本地保持独立决策；云端断网、超时或返回无效结果时退回本地策略。

```text
飞行状态、测距 → 四帧历史 → GRU-IQN → ART / CVaR → 动作提议
       └─ 同步RGB图像 → 云端VLM → 风险校验与有界融合 ──┘
```

**当前提供可运行的上位机影子模式：输出动作提议，不向无人机发送控制指令。不是下载后即可自主起飞的固件。**真实 Qwen/LoRA 效果、融合收益与实机闭环仍需分别验证。

## 目录

| 目录 | 内容 |
|---|---|
| `navigation/` | 本地 GRU-IQN、ART、异步云端客户端、历史状态和运行入口 |
| `inference/` | 原云端离线推理及新增在线风险服务 |
| `examples/` | 不含真实飞行数据的 JSONL 输入格式示例 |
| `pipeline/`、`training/` | 保留原云端数据处理、LoRA训练流程 |
| `docs/` | 接口协议、原云端复现指南和方法说明 |

## 1. 下载与安装

```bash
git clone https://github.com/2fhkf3/Cloud-based-Semantic-Risk-Perception.git
cd Cloud-based-Semantic-Risk-Perception
python -m pip install -r requirements-navigation.txt
```

使用 Python 3.10+。本地导航可以使用 CPU；CUDA 用户应安装与设备匹配的 PyTorch。云端服务可在另一台机器及独立 Python 环境运行，不要求本地安装大模型依赖。

## 2. 准备导航权重并运行

已附带基线权重 `models/navigation/IQN_best_gru_dagger_v3.pth`（约8.6MB），输入为4帧×20维，隐藏特征512维，输出12个离散动作。安装依赖后可直接执行下面的命令，不需要先训练模型。该权重不是随机初始化模型，也不是未验收的实验调速候选。

已核对的基线 SHA-256：

```text
002c9cc35869d8fd6c77d5be4b18a248f89ba59d6aef9dabceb2e831829828a6
```

```bash
python -m navigation.run --input examples/observations.jsonl --checkpoint models/navigation/IQN_best_gru_dagger_v3.pth --expected-sha256 002c9cc35869d8fd6c77d5be4b18a248f89ba59d6aef9dabceb2e831829828a6 --output artifacts/local_run_001.jsonl
```

示例是人为构造的静止状态，用于跑通输入输出，不是仿真轨迹、导航成功率或真实飞行结果。输出包含动作ID、速度、ART/fused alpha、观测时效及权重哈希。再次运行需换一个输出文件名，程序不会覆盖已有记录。

## 3. 启动真实云端服务

在云端先安装与 CUDA 或 Ascend/CANN 匹配的 PyTorch / torchvision；Ascend 还需匹配的 torch_npu。已有云端环境沿用原加速器配置，不要用桌面端依赖替换。原 Ascend 配置及训练步骤见[云端复现指南](docs/cloud_reproduction.md)。

```bash
python -m pip install -r requirements-service.txt
python -m inference.serve --model /path/to/Qwen2.5-VL-3B-Instruct --adapter /path/to/LoRA-checkpoint --host 127.0.0.1 --port 8080
```

Qwen基座模型和LoRA适配器需要自行准备，不随代码下载。服务复用原推理实现，模型常驻内存；需要可用的CUDA或Ascend设备。`/healthz` 只说明服务已加载，不代表模型风险判断准确。

跨机器部署时设置 `SEMANTIC_API_TOKEN`，通过HTTPS反向代理暴露服务，再配置监听地址。令牌不要提交到GitHub；本服务不应直接裸露到公网。

## 4. 连接云端

在自己的JSONL每行增加 `image_path`、`frame_timestamp_s`，图像路径相对输入文件所在目录。示例输入没有RGB，不会发起VLM请求。真实请求须使用同步采集的PNG/JPEG图像，图像与状态时间差不超过100ms；每帧路径在异步请求完成前不得被覆盖。

```bash
python -m navigation.run --input /path/to/synchronized_observations.jsonl --pace --checkpoint models/navigation/IQN_best_gru_dagger_v3.pth --cloud-endpoint http://127.0.0.1:8080/v1/semantic-risk --cloud-interval 1 --risk-ttl 2 --output artifacts/cloud_run_001.jsonl
```

远端必须改用HTTPS。`--input -` 可读取实时标准输入，此时观测和图像时间戳必须与消费端使用同一本机 `time.monotonic()` 时钟；文件重放使用虚拟记录时间，不能用来报告真实端到端时延。

## 5. 输入、任务切换与执行边界

- 云端7维状态：`[x,y,d_goal,d_front,d_left,d_back,d_right]`，单位米；`yaw_rad` 是额外字段，不是第8维状态。
- 本地每帧20维：相对目标2维、实际速度2维、四向测距4维、上一实际动作one-hot12维。
- 动作：4方向×3档速度，约0.23024 / 0.55156 / 1.0 m/s；固定yaw训练约定，朝向不符时HOLD。
- 风险通过置信度、会话和时效校验后，只能降低当前ART alpha，不能直接输出电机指令。
- 切换任务须插入 `{"type":"start_session","session_id":"新的唯一任务ID"}`，不能通过不同ID的传感器包自动切换。旧ID不能复用。
- 图像和HTTP在异步工作进程中处理；过期观测输出HOLD。影子提议不算实际动作，也不更新ART执行状态。

完整格式及执行器接入约定见[接口协议](docs/integration_protocol.md)。输入中的 `previous_executed_action_id` 必须来自实际执行记录，不能把未执行的模型预测写入历史。

板端固件、TVM/GAP8编译、通信协议升级及真实飞控执行不属于本次运行入口。后续接实机还需独立执行仲裁、下发时效检查、硬件看门狗及起降流程。语义风险分数不是经过校准的碰撞概率。
