import math
import os

import torch
import torch.distributed as dist


__LOG10 = math.log(10)


def mse2psnr(x):
    return -10.*torch.log(x)/__LOG10


def init_ddp():
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        return 0, 1  # Single GPU run
        
    dist.init_process_group(backend="nccl")
    print(f'Initialized process {local_rank} / {world_size}')
    torch.cuda.set_device(local_rank)

    setup_dist_print(local_rank == 0)
    return local_rank, world_size


def setup_dist_print(is_main):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def using_dist():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not using_dist():
        return 1
    return dist.get_world_size()


def get_rank():
    if not using_dist():
        return 0
    return dist.get_rank()


def gather_all(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]

    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    return tensor_list


def reduce_dict(input_dict, average=True):
    """
    Reduces the values in input_dict across processes, when distributed computation is used.
    In all processes, the dicts should have the same keys mapping to tensors of identical shape.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    keys = sorted(input_dict.keys())
    values = [input_dict[k] for k in keys] 

    if average:
        op = dist.ReduceOp.AVG
    else:
        op = dist.ReduceOp.SUM

    for value in values:
        dist.all_reduce(value, op=op)

    reduced_dict = {k: v for k, v in zip(keys, values)}

    return reduced_dict
