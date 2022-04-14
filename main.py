import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="DCGAN", help= "Model types: (DCGAN | WGAN) ")
parser.add_argument("-e", "--epochs", type=int, default=200, help="Passes through entire dataset (DCGAN only)")
parser.add_argument("-b", "--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("-lr", "--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("-c", "--clip_value", type=float, default=0.01, help="Gradient Clipping")
parser.add_argument("-i", "--iterations", type=int, default=40000, help="Iterations for WGAN")
args = parser.parse_args()

if args.model == "DCGAN":
    print(f"Training {args.model} for {args.epochs} epochs")
    DCGAN(
        epochs=args.epochs,
        batch_size=args.batch_size, 
        lr=args.lr
    )
elif args.model == "WGAN":
    print(f"Training {args.model} for {args.iterations} iterations")
    WGAN(
        clip_value=args.clip_value,
        iterations=args.iterations, 
        batch_size=args.batch_size, 
        lr=args.lr,
    )
elif args.model == "ACGAN":
    print(f"Training {args.model} for {args.epochs} epochs")
    ACGAN(
        epochs=args.epochs,
        batch_size=args.batch_size, 
        lr=args.lr,
        crop_shape=32,
        in_size=110
    )
