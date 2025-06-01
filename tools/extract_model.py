# Help extract model state from checkpoint file

import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Extract model state from checkpoint')
    parser.add_argument('--checkpoint',
                       required=True,
                       help='path to full checkpoint file')
    parser.add_argument('--output',
                       required=True,
                       help='path to save extracted model state')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load full checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Extract only model state
    model_state = checkpoint['model']
    
    # Save model state
    torch.save(model_state, f"{args.output}.pth")
    print(f"Model state extracted and saved to {args.output}")

if __name__ == '__main__':
    main()