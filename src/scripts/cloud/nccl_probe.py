import datetime
import os
import torch
import torch.distributed as dist

if __name__ == '__main__':
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=90))
    x=torch.tensor([rank+1.],device=f'cuda:{rank}')
    dist.all_reduce(x)
    assert x.item()==3
    dist.barrier()
    dist.destroy_process_group()
