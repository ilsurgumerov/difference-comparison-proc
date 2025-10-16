# import os
# import torch
# import torch.distributed as dist
# from .common import init_dist

# ELL = 31
# MOD = 1 << ELL  # модуль для арифметики

# def compare_protocol(rank, world_size):
#     """
#     Надёжная реализация: разделяем a,b на доли, обмениваемся так, чтобы избежать deadlock,
#     генерируем случайные доли корректно и печатаем x,y.
#     """
#     # init
#     init_dist(rank, world_size)
#     print(f"[rank{rank}] init_dist done", flush=True)

#     if rank == 1:
#         # A (owener a)
#         a = int(os.environ.get("A_VALUE", "5"))

#         # generate a1,a2 : a1 + a2 = a
#         a1 = torch.randint(0, MOD, (1,), dtype=torch.long)
#         a2 = torch.tensor([(a - int(a1.item())) % MOD], dtype=torch.long)

#         print(f"[A] a={a}, a1={int(a1.item())}, a2={int(a2.item())}", flush=True)

#         dist.send(tensor=a2, dst=2)

#         b1 = torch.zeros(1, dtype=torch.long)
#         dist.recv(tensor=b1, src=2)
#         print(f"[A] received b1={int(b1.item())} from rank2", flush=True)

#         # find x 
#         x = (a1 - b1) % MOD
#         print(f"[A] x = (a1 - b1) % M = {int(x.item())}", flush=True)

#     elif rank == 2:
#         # B (owener b)
#         b = int(os.environ.get("B_VALUE", "10"))

#         # generate b1,b2: b1 + b2 = b
#         b2 = torch.randint(0, MOD, (1,), dtype=torch.long)
#         b1 = torch.tensor([(b - int(b2.item())) % MOD], dtype=torch.long)

#         print(f"[B] b={b}, b1={int(b1.item())}, b2={int(b2.item())}", flush=True)

#         a2 = torch.zeros(1, dtype=torch.long)
#         dist.recv(tensor=a2, src=1)
#         print(f"[B] received a2={int(a2.item())} from rank1", flush=True)

#         dist.send(tensor=b1, dst=1)

#         # find y
#         y = (a2 - b2) % MOD
#         print(f"[B] y = (a2 - b2) % M = {int(y.item())}", flush=True)

#     else:
#         # rank 0 (TTP)
#         print("[TTP] Trusted Third Party (пока ничего не делает)", flush=True)


# tasks/mpc_compare.py
import torch
import torch.distributed as dist
from .common import init_dist
from .state import PartyState
from .phase1_share_diff import phase1

ELL = 31  

def compare_protocol(rank, world_size):
    init_dist(rank, world_size)
    print(f"[rank{rank}] init_dist done", flush=True)

    state = PartyState(rank, world_size, ELL)

    phase1(state)