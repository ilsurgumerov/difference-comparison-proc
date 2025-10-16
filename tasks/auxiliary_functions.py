import os
import torch
import torch.distributed as dist

def request_beaver_triple(ell=31):
    """
    Запрос одной тройки Бивера от TTP (rank 0).
    Возвращает (a_share, b_share, c_share)
    """
    REQ_TAG = 100
    TRIPLE_TAG = 200

    req = torch.tensor(ell, dtype=torch.long)
    dist.send(req, dst=0, tag=REQ_TAG)  # запрос

    triple = torch.zeros(3 * ell, dtype=torch.long)
    dist.recv(triple, src=0, tag=TRIPLE_TAG)

    a_share = triple[:ell]
    b_share = triple[ell:2 * ell]
    c_share = triple[2 * ell:]

    return a_share, b_share, c_share


def int_to_twos_complement_bits(value: int, ell: int):
    """
    Преобразует число в двоичное представление (дополнительный код) длиной ELL бит.
    Первый бит — знак (0 = +, 1 = -).
    """
    total_bits = ell

    # Для отрицательных — представление через дополнительный код
    if value < 0:
        value = (1 << total_bits) + value 

    # Получаем биты от старшего к младшему
    bits = torch.tensor([(value >> i) & 1 for i in reversed(range(total_bits))], dtype=torch.long)
    return bits

def get_terms(state):
    """
    Арифметическое разделение входов и получение локальных долей разности 
    rank1 берет просто свое значение a : term == a
    rank2 берет для вычисления суммы -b: term == -b
    (a - b) = term(rank1) + term(rank2)
    """
    rank = state.rank
    
    if rank == 1:
        a = int(os.environ.get("A_VALUE", "5"))
  
        state.term = torch.tensor(a)
        print(f"[A] A stored = {int(state.term.item())}", flush=True)

    elif rank == 2:
        b = int(os.environ.get("B_VALUE", "10"))
        
        state.term = torch.tensor(-b)
        print(f"[B] B stored = {int(state.term.item())}", flush=True)

    else:
        print("[TTP] phase1 idle", flush=True)
        return None


def bitwise_decomposition_of_shares(state):
    """
    Побитовое разложение локальных долей x,y и раздача битовых шейров
    Из term в десятичном представлении -> term1, teram2 - в двоичном
    term1 - x, term2 = y; у каждого воркера свой:
    term1(rank1) XOR term2(rank2) == term(rank1)
    term1(rank2) XOR term2(rank2) == term(rank2)
    """
    rank = state.rank

    term = state.term
    term_val = int(term.item())
    # bits tensor
    term_bits = int_to_twos_complement_bits(term,  state.ell)

    state.term_bits = term_bits
    
    term_rand_bit = torch.randint(0, 2, ((state.ell),), dtype=torch.long)
    term_loc_bit = term_bits ^ term_rand_bit

    # shared bit
    term_received = torch.zeros((state.ell), dtype=torch.long)

    if rank == 1:
        dist.send(term_rand_bit, dst=2)
        dist.recv(term_received, src=2)

        state.term1 = term_loc_bit
        state.term2 = term_received

    elif rank == 2:
        dist.recv(term_received, src=1)
        dist.send(term_rand_bit, dst=1)
            
        state.term1 = term_received
        state.term2 = term_loc_bit
    else:
        print("[TTP] phase2 idle", flush=True)
        return None

def bitwise_secret_sharing_and(state):
    """
    Разделение секрета побитового И term(rank1) & teram(rank2)
    Возвращает: для каждого ранга свою часть разделения секрета
    """

    rank = state.rank

    a_loc, b_loc, c_loc = request_beaver_triple(state.ell)
    d_loc = state.term1 ^ a_loc
    e_loc = state.term2 ^ b_loc

    d_shared = torch.zeros((state.ell), dtype=torch.long)
    e_shared = torch.zeros((state.ell), dtype=torch.long)

    if rank == 1:
        dist.send(d_loc, dst=2)
        dist.recv(d_shared, src=2)

        dist.send(e_loc, dst=2)
        dist.recv(e_shared, src=2)
    elif rank == 2:
        dist.recv(d_shared, src=1)
        dist.send(d_loc, dst=1)

        dist.recv(e_shared, src=1)
        dist.send(e_loc, dst=1)
    else:
        print("[TTP] phase3 idle", flush=True)
        return None

    d = d_shared ^ d_loc
    e = e_shared ^ e_loc

    print(f"d = {d}, \ne = {e}")

    z = c_loc ^ (d & b_loc) ^ (e & a_loc)

    if rank == 1:
        z = z ^ (d & e)
    
    # set in state
    state.term_shared_and = z


def find_new_p(rank, p_prev, xXORy_i, xANDy_i):
    a_loc, b_loc, c_loc = request_beaver_triple(ell=1)
    d_loc = xXORy_i ^ a_loc
    e_loc = p_prev ^ b_loc

    d_shared = torch.zeros((1), dtype=torch.long)
    e_shared = torch.zeros((1), dtype=torch.long)

    if rank == 1:
        dist.send(d_loc, dst=2)
        dist.recv(d_shared, src=2)

        dist.send(e_loc, dst=2)
        dist.recv(e_shared, src=2)
    elif rank == 2:
        dist.recv(d_shared, src=1)
        dist.send(d_loc, dst=1)

        dist.recv(e_shared, src=1)
        dist.send(e_loc, dst=1)
    else:
        print("[TTP] phase3 idle", flush=True)
        return None

    d = d_shared ^ d_loc
    e = e_shared ^ e_loc

    t = c_loc ^ (d & b_loc) ^ (e & a_loc)

    if rank == 1:
        t ^= (d & e)

    p_i = xANDy_i ^ t

    return p_i


def find_z(state):
    rank = state.rank
    xXORy = state.term_shared_xor
    xANDy = state.term_shared_and
    
    p = torch.zeros(state.ell + 1, dtype=torch.long)

    for i in range(state.ell - 1, -1, -1):
        p_i = find_new_p(rank, p[i + 1], xXORy[i], xANDy[i])
        p[i] = p_i
    
    return (xXORy ^ p[1:])
