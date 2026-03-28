import os
import subprocess


def get_free_gpus(memory_threshold_mb=500, util_threshold=10):
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip().splitlines()

    free = []
    for line in out:
        idx, mem, util = [x.strip() for x in line.split(",")]
        idx = int(idx)
        mem = int(mem)
        util = int(util)

        if mem < memory_threshold_mb and util < util_threshold:
            free.append(idx)

    return free


def set_visible_gpus(memory_threshold_mb=500, util_threshold=10):
    free = get_free_gpus(
        memory_threshold_mb=memory_threshold_mb,
        util_threshold=util_threshold
    )

    if free:
        visible = ",".join(map(str, free))
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
        print(f"[auto_gpu] Using free GPUs: {visible}")
    else:
        # Force CPU by hiding all GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[auto_gpu] No free GPUs found. Forcing CPU.")

    return free