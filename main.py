import argparse
import sys
import os
import torch
import cv2
import time
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import load_config, get_device, setup_carla_env
from training.train_stage1 import train_ae
from training.train_stage2 import train_ctrl
from training.extract_features import extract_features
from inference.predictor import Predictor
from carla_integration.carla_client import CarlaClient
from carla_integration.sensor_manager import SensorManager

def run_drive(args, cfg):
    """Runs autonomous driving in CARLA simulator."""
    setup_carla_env(cfg)
    device = get_device(cfg['device'])
    
    predictor = Predictor(args.ae_path, args.ctrl_path, device, cfg)
    
    # Simulation settings from config
    c_cfg = cfg['carla_env']
    carla_client = CarlaClient(c_cfg['host'], c_cfg['port'], c_cfg['timeout'])
    carla_client.set_weather(cfg['simulation']['weather'])
    
    vehicle = carla_client.spawn_vehicle(cfg['simulation']['vehicle_model'])
    sensor_manager = SensorManager(carla_client.world, vehicle, cfg)
    
    print("[INFO] AI Control initialized. Press 'Q' or Ctrl+C to exit.")
    
    try:
        import carla
        while True:
            carla_image = sensor_manager.get_latest_image()
            if carla_image is None:
                continue
                
            image_array = sensor_manager.process_image(carla_image)
            steer, throttle, brake = predictor.predict(image_array)
            
            control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
            vehicle.apply_control(control)
            
            # Simple visualization
            display = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            status = f"Steer:{steer:.2f} Throttle:{throttle:.2f} Brake:{brake:.2f}"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("CARLA AI Driving Pipeline", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Manual interrupt detected.")
    except Exception as e:
        print(f"[ERROR] Simulation error: {e}")
    finally:
        carla_client.cleanup()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Autonomous Driving Production Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["train_ae", "extract_features", "train_ctrl", "drive"], 
                        help="Operation mode")
    
    # Optional arguments for checkpoints and data paths
    parser.add_argument("--ae_path", type=str, help="Path to AutoEncoder checkpoint")
    parser.add_argument("--ctrl_path", type=str, help="Path to Controller checkpoint")
    parser.add_argument("--labels_csv", type=str, help="Path to raw labels CSV for feature extraction")
    parser.add_argument("--latent_csv", type=str, help="Path to extracted latent features CSV")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        return
        
    cfg = load_config(args.config)
    
    if args.mode == "train_ae":
        train_ae(args.config)
        
    elif args.mode == "extract_features":
        if not args.ae_path or not args.labels_csv:
            print("[ERROR] --ae_path and --labels_csv are required for extraction.")
            return
        output_csv = cfg['paths']['latent_data_csv']
        extract_features(args.config, args.ae_path, args.labels_csv, output_csv)
        
    elif args.mode == "train_ctrl":
        latent_path = args.latent_csv if args.latent_csv else cfg['paths']['latent_data_csv']
        if not os.path.exists(latent_path):
            print(f"[ERROR] Latent data not found: {latent_path}")
            return
        train_ctrl(args.config, args.ae_path, latent_path)
        
    elif args.mode == "drive":
        if not args.ae_path or not args.ctrl_path:
            print("[ERROR] --ae_path and --ctrl_path are required for driving mode.")
            return
        run_drive(args, cfg)

if __name__ == "__main__":
    main()
