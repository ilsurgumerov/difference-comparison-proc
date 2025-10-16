import os
import torch
import torch.distributed as dist
from .common import init_dist
from .state import PartyState
from .auxiliary_functions import get_terms, bitwise_decomposition_of_shares, bitwise_secret_sharing_and, find_z

ELL = int(os.environ.get("MAX_DIGIT", "15")) + 1

def compare_protocol(rank, world_size):
    init_dist(rank, world_size)
    print(f"[rank{rank}] init_dist done", flush=True)

    state = PartyState(rank, world_size, ELL)

    get_terms(state) # state.term
    
    bitwise_decomposition_of_shares(state) # state.term1, state.term2

    bitwise_secret_sharing_and(state) # state.term_shared_and
    
    state.term_shared_xor = state.term1 ^ state.term2

    z = find_z(state)  # term1(ranki) + term2(ranki) in binary format
    
    z_received = torch.zeros_like(z)

    if state.rank == 1:
        dist.send(z, dst=2)
        dist.recv(z_received, src=2)

    elif state.rank == 2:
        dist.recv(z_received, src=1)
        dist.send(z, dst=1)

    sum_result = z ^ z_received # term(rank1) + term(rank2) in binary format
    
    print(f"Source (a - b) = x + y result in bin form: {sum_result}")
    print(f"a - b < 0 ? : {sum_result[0]}")




