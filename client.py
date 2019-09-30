
import argparse
import os
import threading
import time
import torch
import torch.distributed as dist


import queue
import threading
import time
import argparse
import time
from sys import platform
import torch

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
global exitFlag
exitFlag=[False]

import os





NUM_TRIALS = 20

def receive_tensor_helper(tensor, src_rank, group, tag, num_iterations,
                          intra_server_broadcast):
    for i in range(num_iterations):
        if intra_server_broadcast:
            dist.broadcast(tensor=tensor, group=group, src=src_rank)
        else:
            dist.recv(tensor=tensor, src=src_rank, tag=tag)
    print("Done with tensor size")

def send_tensor_helper(tensor, dst_rank, group, tag, num_iterations,
                       intra_server_broadcast):
    for i in range(num_iterations):
        if intra_server_broadcast:
            dist.broadcast(tensor=tensor, group=group, src=1-dst_rank)
        else:
            dist.send(tensor=tensor, dst=dst_rank, tag=tag)
    print("Done with tensor size" )

def start_helper_thread(func, args):
    helper_thread = threading.Thread(target=func,
                                     args=tuple(args))
    helper_thread.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test lightweight communication library')
    parser.add_argument("--backend", type=str, default='gloo',
                        help="Backend")
    parser.add_argument('-s', "--send", action='store_true',
                        help="Send tensor (if not specified, will receive tensor)")
    parser.add_argument("--master_addr", type=str,default='localhost',
                        help="IP address of master")
    parser.add_argument("--use_helper_threads", action='store_true',
                        help="Use multiple threads")
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank of current worker")
    parser.add_argument('-p', "--master_port", default=12445,
                        help="Port used to communicate tensors")
    parser.add_argument("--intra_server_broadcast", action='store_true',
                        help="Broadcast within a server")

    args = parser.parse_args()

    TKD_decoder = Darknet('cfg/TKD_decoder.cfg')

    TKD_decoder.load_state_dict(torch.load('weights/TKD.pt')['model'])
    TKD = TKD_decoder

    num_ranks_in_server = 1
    if args.intra_server_broadcast:
        num_ranks_in_server = 2
    local_rank = args.rank % num_ranks_in_server
    torch.cuda.set_device(local_rank)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    world_size = 2
    dist.init_process_group(args.backend, rank=args.rank, world_size=world_size)


    TKD=TKD.cuda()
    for i in range(1000):
        time.sleep(1)

        tensor = torch.ones([1, 3, 416, 416]).cpu()

        send_tensor_helper(tensor, 1-args.rank, 0, 0,
                           1, args.intra_server_broadcast)
        for parm in TKD.parameters():
            temp=parm.clone()
            temp=temp.cpu()
            receive_tensor_helper(temp, 1 - args.rank, 0, 0,
                                  1, args.intra_server_broadcast)
            temp=torch.zeros(temp.shape)
            parm[:]=temp.cuda()




