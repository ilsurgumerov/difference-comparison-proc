import torch
import torch.distributed as dist
from tasks.common import init_dist

def generate_beaver_triple(tensor_size):
    """
    Генерация одной тройки Бивера длиной ell бит.
    Возвращает два тензора (для rank1 и rank2).
    """
    a = torch.randint(0, 2, (tensor_size,), dtype=torch.long)
    b = torch.randint(0, 2, (tensor_size,), dtype=torch.long)
    c = a & b

    a1 = torch.randint(0, 2, (tensor_size,), dtype=torch.long)
    a2 = a ^ a1
    b1 = torch.randint(0, 2, (tensor_size,), dtype=torch.long)
    b2 = b ^ b1
    c1 = torch.randint(0, 2, (tensor_size,), dtype=torch.long)
    c2 = c ^ c1

    triple_rank1 = torch.cat([a1, b1, c1])
    triple_rank2 = torch.cat([a2, b2, c2])

    return triple_rank1, triple_rank2


def run_ttp(rank, world_size):
    """
    Trusted Third Party (TTP):
    слушает запросы от участников (rank 1 и 2),
    генерирует тройки Бивера и рассылает доли.
    """
    init_dist(rank, world_size)
    print(f"[rank{rank}] init_dist done", flush=True)
    print("[TTP] Beaver triple generator started", flush=True)

    REQ_TAG = 100
    TRIPLE_TAG = 200

    try:
        while True:
            # Ждём запросы от обеих сторон (должно придти число - вектор размера троек бивера)
            beaver_count1 = torch.zeros(1, dtype=torch.long)
            beaver_count2 = torch.zeros(1, dtype=torch.long)

            print("[TTP] Waiting for requests...", flush=True)
            dist.recv(beaver_count1, src=1, tag=REQ_TAG)
            dist.recv(beaver_count2, src=2, tag=REQ_TAG)

            if (beaver_count1 != beaver_count2):
                print("[TTP] Error: Beaver triple different size", flush=True)
                return None

            # Сгенерировать тройку (битовые строки в виде тензоров пришедшего размера)
            t1, t2 = generate_beaver_triple(int(beaver_count1.item()))
            print(f"[TTP] Generated Beaver triple (len={int(beaver_count1.item())})", flush=True)

            # Отправить доли
            dist.send(t1, dst=1, tag=TRIPLE_TAG)
            dist.send(t2, dst=2, tag=TRIPLE_TAG)
            print("[TTP] Sent triple shares to rank1 & rank2", flush=True)

    except RuntimeError as e:
        if "Connection closed by peer" in str(e):
            print("[TTP] All workers disconnected — shutting down gracefully.", flush=True)
        else:
            print(f"[TTP] Unexpected error: {e}", flush=True)
            raise

    finally:
        dist.destroy_process_group()
        print("[TTP] Process group destroyed. Exit cleanly.", flush=True)