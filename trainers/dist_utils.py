"""
Distributed training (DDP, DistributedDataParallel) utilities.

DDP launches one process per GPU (each with its own dataloader workers)
and trains far more evenly and quickly. Launch with torchrun:

    torchrun --nproc_per_node=8 trainers/classification_trainer.py --config-path ...

torchrun sets the RANK/WORLD_SIZE/LOCAL_RANK environment variables;
without them (= launched normally with python) it automatically runs as a single process.
"""
import os
from datetime import timedelta
import torch
import torch.distributed as dist


def _dist_ready():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if _dist_ready() else 0


def get_world_size():
    return dist.get_world_size() if _dist_ready() else 1


def is_main_process():
    # Identify rank 0 so that saving/logging/wandb run only in that process
    return get_rank() == 0


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


def distributed_launched():
    # Check whether multiple processes were launched by torchrun
    return (
        'RANK' in os.environ
        and 'WORLD_SIZE' in os.environ
        and int(os.environ.get('WORLD_SIZE', '1')) > 1
    )


def setup_distributed():
    """
    Prepare the distributed environment for the launch mode and choose the device this process uses.
    Returns: (distributed: bool, device: torch.device, local_rank: int)
    """
    if distributed_launched() and torch.cuda.is_available():
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        # NCCL: communication backend between NVIDIA GPUs
        # Set a generous timeout (1 hour). The final test runs only on rank 0 while the other ranks
        #  wait at a barrier; with a large test set this can exceed the default NCCL timeout (10-30 min depending
        #  on version) and kill the barrier, so extra headroom is given.
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=1))
        return True, torch.device('cuda', local_rank), local_rank

    # Not distributed: single device (cuda if a GPU is available, otherwise cpu)
    if torch.cuda.is_available():
        return False, torch.device('cuda'), 0
    return False, torch.device('cpu'), 0


def barrier():
    # Wait until all processes reach this point (synchronisation)
    if _dist_ready():
        dist.barrier()


def broadcast_bool(flag, device):
    # Broadcast a bool decision computed only on rank 0 (e.g. whether to early-stop) to all processes.
    #  Under DDP all ranks must follow the same flow (forward/backward/scheduler) every epoch, so if
    #  only rank 0 stops training, the other ranks hang waiting for the next communication.
    if not _dist_ready():
        return flag
    tensor = torch.tensor([1 if flag else 0], device=device)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())


def cleanup_distributed():
    if _dist_ready():
        dist.barrier()
        dist.destroy_process_group()
