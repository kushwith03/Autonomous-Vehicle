import argparse
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train_stage1 import train_stage1
from training.train_cnn_bc import train_cnn_bc
from training.evaluate import evaluate_controller
from training.extract_features import extract_features
from training.train_mlp_ctrl import train_mlp_ctrl
from inference.predictor import Predictor
from utils.benchmark import benchmark_inference, print_benchmark_report, assert_latency_target

def drive(config_path, ae_path, ctrl_path):
    print("🚀 Starting Autonomous Drive Mode (Simulation)...")
    print(f"AE: {ae_path}")
    print(f"Controller: {ctrl_path}")
    # In a real scenario, this would initialize CARLA and run the predictor in a loop.
    # Since we don't have CARLA here, we'll simulate a start.
    predictor = Predictor(ae_path, ctrl_path)
    print("✅ Predictor initialized. Ready for CARLA connection.")

def main():
    parser = argparse.ArgumentParser(description="Autonomous Vehicle Pipeline")
    parser.add_argument("--mode", type=str, choices=["train_ae", "extract_features", "train_ctrl", "train_cnn_bc", "drive", "benchmark", "evaluate"], required=True, help="Mode to run")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--ae_path", type=str, help="Path to autoencoder checkpoint")
    parser.add_argument("--ctrl_path", type=str, help="Path to controller checkpoint")
    parser.add_argument("--latent_csv", type=str, help="Path to latent CSV for evaluation")
    
    args = parser.parse_args()

    if args.mode == "train_ae":
        train_stage1(args.config)
    elif args.mode == "extract_features":
        extract_features(args.config)
    elif args.mode == "train_ctrl":
        train_mlp_ctrl(args.config)
    elif args.mode == "train_cnn_bc":
        train_cnn_bc(args.config)
    elif args.mode == "drive":
        if not args.ae_path or not args.ctrl_path:
            print("Error: --ae_path and --ctrl_path are required for drive mode.")
            sys.exit(1)
        drive(args.config, args.ae_path, args.ctrl_path)
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
    elif args.mode == "evaluate":
        if not args.ctrl_path or not args.latent_csv:
            print("Error: --ctrl_path and --latent_csv are required for evaluate mode.")
            sys.exit(1)
        
        evaluate_controller(args.config, args.latent_csv, args.ctrl_path)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
