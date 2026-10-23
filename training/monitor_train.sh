#!/usr/bin/env bash
# 全量训练监控：每 5 分钟向 logs/full_train_monitor.log 追加一行状态。
# 训练进程消失时记录最终状态（正常完成 / 异常退出）后退出。
# 用 setsid 启动，脱离终端会话：setsid nohup bash monitor_train.sh >/dev/null 2>&1 &

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"
OUT="${LOG_DIR}/full_train_monitor.log"

timestamp() { date '+%F %T'; }

echo "[$(timestamp)] monitor started (pid $$)" >> "${OUT}"

while true; do
    tlog="$(ls -t "${LOG_DIR}"/train_*.log 2>/dev/null | head -1)"
    if [[ -z "${tlog}" ]]; then
        echo "[$(timestamp)] ERROR: no train_*.log found" >> "${OUT}"
        exit 1
    fi

    if ! pgrep -f "python.*swift/cli/sft\.py" >/dev/null 2>&1; then
        if tr '\r' '\n' < "${tlog}" | grep -q "Training script finished successfully"; then
            echo "[$(timestamp)] FINISHED OK — training completed successfully (${tlog})" >> "${OUT}"
        else
            echo "[$(timestamp)] FAILED — training process is gone but no success marker in ${tlog}; last lines:" >> "${OUT}"
            tr '\r' '\n' < "${tlog}" | tail -5 | sed 's/^/    /' >> "${OUT}"
        fi
        exit 0
    fi

    metrics="$(tr '\r' '\n' < "${tlog}" | grep -oE "\{'(loss|eval_loss)[^}]*\}" | tail -40)"
    step="$(echo "${metrics}" | grep -oE "global_step/max_steps': '[0-9]+/[0-9]+'" | tail -1 | grep -oE "[0-9]+/[0-9]+")"
    loss="$(echo "${metrics}" | grep "'loss':" | tail -1 | grep -oE "'loss': '[0-9.]+' | 'token_acc': '[0-9.]+'" | sed "s/'//g")"
    eval_loss="$(echo "${metrics}" | grep "'eval_loss':" | tail -1 | grep -oE "'eval_loss': '[0-9.]+'" | sed "s/'//g")"
    eta="$(echo "${metrics}" | grep -oE "'remaining_time': '[^']*'" | tail -1 | sed "s/'//g")"
    mem="$(echo "${metrics}" | grep -oE "'memory\(GiB\)': '[0-9.]+'" | tail -1 | sed "s/'//g")"
    npu="$(npu-smi info 2>/dev/null | awk '/^\| [4-6] +910B3/{id=$2} /0000:/ && id {v=$(NF-2); gsub(/\//,"",v); printf "%s:%sMB ", id, v; id=""}')"

    echo "[$(timestamp)] step ${step:-?} | ${loss:-loss=?} | ${eval_loss:-no-eval-yet} | ${eta:-eta=?} | ${mem:-mem=?} | NPU4-6 HBM: ${npu:-?}" >> "${OUT}"
    sleep 300
done
