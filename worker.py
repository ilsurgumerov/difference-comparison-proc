import argparse
from importlib import import_module

def main():
    parser = argparse.ArgumentParser(prog="worker")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--target", required=True, help="Напр. tasks.mpc_compare:compare_protocol")
    args = parser.parse_args()

    module_path, func_name = args.target.split(":")
    fn = getattr(import_module(module_path), func_name)
    fn(rank=args.rank, world_size=args.world_size)

if __name__ == "__main__":
    main()
