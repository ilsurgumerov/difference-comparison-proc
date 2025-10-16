# tasks/phase1_share_diff.py
import os
import torch
import torch.distributed as dist

def phase1(state):
    """
    Фаза 1: арифметическое разделение входов и получение локальных долей разности x, y
    (state модифицируется: state['x'] или state['y'] записывается как torch.tensor(1,))
    """
    rank = state.rank
    MOD = state.mod

    if rank == 1:
        a = int(os.environ.get("A_VALUE", "5")) % MOD
        a1 = torch.randint(0, min(MOD, 1 << 30), (1,), dtype=torch.long)
        a2 = torch.tensor([(a - int(a1.item())) % MOD], dtype=torch.long)

        print(f"[A] a={a}, a1={int(a1.item())}, a2={int(a2.item())}", flush=True)

        # send a2 -> rank2
        dist.send(tensor=a2, dst=2)
        print("[A] sent a2 -> rank2", flush=True)

        # receive b1 <- rank2
        b1 = torch.zeros(1, dtype=torch.long)
        dist.recv(tensor=b1, src=2)
        print(f"[A] received b1={int(b1.item())} from rank2", flush=True)

        x = (a1 - b1) % MOD
        state.set("xy", x)
        print(f"[A] x stored = {int(x.item())}", flush=True)

    elif rank == 2:
        b = int(os.environ.get("B_VALUE", "10")) % MOD
        b2 = torch.randint(0, min(MOD, 1 << 30), (1,), dtype=torch.long)
        b1 = torch.tensor([(b - int(b2.item())) % MOD], dtype=torch.long)

        print(f"[B] b={b}, b1={int(b1.item())}, b2={int(b2.item())}", flush=True)

        a2 = torch.zeros(1, dtype=torch.long)
        dist.recv(tensor=a2, src=1)
        print(f"[B] received a2={int(a2.item())} from rank1", flush=True)

        dist.send(tensor=b1, dst=1)
        print("[B] sent b1 -> rank1", flush=True)

        y = (a2 - b2) % MOD
        state.set("xy", y)
        print(f"[B] y stored = {int(y.item())}", flush=True)

    else:
        print("[TTP] phase1 idle", flush=True)

    print("state : ", state.get('xy'))

