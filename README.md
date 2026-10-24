# 云端语义风险辅助的微型无人机自主导航

本项目将云端 Qwen2.5-VL / LoRA 风险感知与本地 GRU-IQN / ART 导航连接起来。云端异步提供语义风险，本地保持独立决策；云端断网、超时或返回无效结果时退回本地策略。

```text
飞行状态、测距 → 四帧历史 → GRU-IQN → ART / CVaR → 动作提议
       └─ 同步RGB图像 → 云端VLM → 风险校验与有界融合 ──┘
```

系统结合GRU时序编码、IQN回报分布估计和ART自适应风险决策，利用历史观测与云端语义信息辅助复杂障碍环境中的导航。仓库提供本地导航策略、云端风险服务、通信接口及运行示例。

## 目录

| 目录 | 内容 |
|---|---|
| `navigation/` | 本地 GRU-IQN、ART、异步云端客户端、历史状态和运行入口 |
| `inference/` | 云端离线推理与在线风险服务 |
| `examples/` | JSONL观测输入示例 |
| `pipeline/`、`training/` | 云端数据处理与LoRA训练流程 |
| `docs/` | 接口协议、云端复现指南和方法说明 |

## 1. 下载与安装

```bash
git clone https://github.com/2fhkf3/Cloud-based-Semantic-Risk-Perception.git
cd Cloud-based-Semantic-Risk-Perception
python -m pip install -r requirements-navigation.txt
```

使用 Python 3.10+。本地导航可以使用 CPU；CUDA 用户应安装与设备匹配的 PyTorch。云端服务可在另一台机器及独立 Python 环境运行，不要求本地安装大模型依赖。

## 2. 运行本地导航

仓库附带导航权重 `models/navigation/IQN_best_gru_dagger_v3.pth`（约8.6MB），输入为4帧×20维，隐藏特征512维，输出12个离散动作。安装依赖后可直接运行示例。

权重 SHA-256：

```text
002c9cc35869d8fd6c77d5be4b18a248f89ba59d6aef9dabceb2e831829828a6
```

```bash
python -m navigation.run --input examples/observations.jsonl --checkpoint models/navigation/IQN_best_gru_dagger_v3.pth --expected-sha256 002c9cc35869d8fd6c77d5be4b18a248f89ba59d6aef9dabceb2e831829828a6 --output artifacts/local_run_001.jsonl
```

该示例使用静止观测序列演示输入格式。命令以影子模式计算并记录动作提议，不下发飞控指令；输出包含动作ID、速度、ART/fused alpha、观测时效及权重哈希。再次运行请更换输出文件名，以保留已有记录。

## 3. 启动云端风险服务

在云端先安装与 CUDA 或 Ascend/CANN 匹配的 PyTorch / torchvision；Ascend 还需匹配的 torch_npu。已有云端环境沿用原加速器配置，不要用桌面端依赖替换。原 Ascend 配置及训练步骤见[云端复现指南](docs/cloud_reproduction.md)。

```bash
python -m pip install -r requirements-service.txt
python -m inference.serve --model /path/to/Qwen2.5-VL-3B-Instruct --adapter /path/to/LoRA-checkpoint --host 127.0.0.1 --port 8080
```

Qwen基座模型和LoRA适配器需要自行准备，不随代码下载。模型常驻内存，服务运行于CUDA或Ascend设备；`/healthz` 用于检查服务就绪状态。

跨机器部署时设置 `SEMANTIC_API_TOKEN`，通过HTTPS反向代理暴露服务，再配置监听地址。令牌不要提交到GitHub；本服务不应直接裸露到公网。

## 4. 连接云端

在自己的JSONL每行增加 `image_path`、`frame_timestamp_s`，图像路径相对输入文件所在目录。示例输入没有RGB，不会发起VLM请求。真实请求须使用同步采集的PNG/JPEG图像，图像与状态时间差不超过100ms；每帧路径在异步请求完成前不得被覆盖。

```bash
python -m navigation.run --input /path/to/synchronized_observations.jsonl --pace --checkpoint models/navigation/IQN_best_gru_dagger_v3.pth --cloud-endpoint http://127.0.0.1:8080/v1/semantic-risk --cloud-interval 1 --risk-ttl 2 --output artifacts/cloud_run_001.jsonl
```

远端使用HTTPS。`--input -` 可读取实时标准输入，此时观测和图像时间戳与消费端使用同一本机 `time.monotonic()` 时钟；文件重放按记录时钟运行。

## 5. 状态与动作定义

- 云端7维状态：`[x,y,d_goal,d_front,d_left,d_back,d_right]`，单位米；`yaw_rad` 是额外字段，不是第8维状态。
- 本地每帧20维：相对目标2维、实际速度2维、四向测距4维、上一实际动作one-hot12维。
- 动作：4方向×3档速度，约0.23024 / 0.55156 / 1.0 m/s；固定yaw训练约定，朝向不符时HOLD。
- 云端语义风险经置信度、会话和时效校验后参与ART alpha调整，本地策略完成动作选择。
- 图像与HTTP请求异步处理，本地导航保持独立运行。

消息格式、任务管理及接口配置见[接口文档](docs/integration_protocol.md)。
