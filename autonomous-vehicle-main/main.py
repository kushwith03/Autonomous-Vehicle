import argparse
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train_stage1 import train_stage1

def main():
    parser = argparse.ArgumentParser(description="Autonomous Vehicle Pipeline")
    parser.add_argument("--mode", type=str, choices=["train_ae"], required=True, help="Mode to run")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    
    args = parser.parse_args()

    if args.mode == "train_ae":
        train_stage1(args.config)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
