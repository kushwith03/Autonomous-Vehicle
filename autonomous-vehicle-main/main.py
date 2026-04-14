import argparse
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train_stage1 import train_stage1
from training.train_cnn_bc import train_cnn_bc
from inference.predictor import Predictor
from utils.benchmark import benchmark_inference, print_benchmark_report, assert_latency_target

def main():
    parser = argparse.ArgumentParser(description="Autonomous Vehicle Pipeline")
    parser.add_argument("--mode", type=str, choices=["train_ae", "train_cnn_bc", "benchmark"], required=True, help="Mode to run")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--ae_path", type=str, help="Path to autoencoder checkpoint")
    parser.add_argument("--ctrl_path", type=str, help="Path to controller checkpoint")
    
    args = parser.parse_args()

    if args.mode == "train_ae":
        train_stage1(args.config)
    elif args.mode == "train_cnn_bc":
        train_cnn_bc(args.config)
    elif args.mode == "benchmark":
        if not args.ae_path or not args.ctrl_path:
            print("Error: --ae_path and --ctrl_path are required for benchmark mode.")
            sys.exit(1)
        
        predictor = Predictor(args.ae_path, args.ctrl_path)
        results = benchmark_inference(predictor)
        print_benchmark_report(results)
        
        try:
            assert_latency_target(results, 50.0)
            print(f"[PASS] Mean inference latency: {results['mean_ms']:.2f}ms < 50ms target")
        except AssertionError as e:
            print(f"[WARN] Latency exceeds 50ms target: {results['mean_ms']:.2f}ms")
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
