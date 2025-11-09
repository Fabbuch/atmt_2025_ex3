#!/usr/bin/env python3
"""
Script to average multiple model checkpoints.

This script takes multiple checkpoint files and averages their parameters
to produce a more stable final model. This is a common technique to reduce
variance in model performance.

Usage:
    # Average the last 5 checkpoints
    python average_checkpoints.py --checkpoint-dir checkpoints/ --num-checkpoints 5 --output averaged_checkpoint.pt

    # Average specific checkpoints
    python average_checkpoints.py --checkpoints checkpoint1.pt checkpoint2.pt checkpoint3.pt --output averaged.pt
"""

import argparse
import os
import sys
import logging
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seq2seq import utils


def get_args():
    parser = argparse.ArgumentParser('Checkpoint Averaging')

    # Option 1: Provide checkpoint directory and select last N
    parser.add_argument('--checkpoint-dir', type=str, help='Directory containing checkpoints')
    parser.add_argument('--num-checkpoints', type=int, default=5,
                        help='Number of last checkpoints to average (default: 5)')
    parser.add_argument('--pattern', type=str, default='checkpoint*.pt',
                        help='Glob pattern to match checkpoint files (default: checkpoint*.pt)')

    # Option 2: Provide specific checkpoint paths
    parser.add_argument('--checkpoints', nargs='+', type=str,
                        help='List of specific checkpoint paths to average')

    # Output
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for averaged checkpoint')

    return parser.parse_args()


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Determine which checkpoints to use
    if args.checkpoints:
        # Use explicitly provided checkpoints
        checkpoint_paths = args.checkpoints
        logging.info(f"Using {len(checkpoint_paths)} explicitly provided checkpoints")

        # Validate that all files exist
        for path in checkpoint_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")

    elif args.checkpoint_dir:
        # Find checkpoints in directory
        if not os.path.isdir(args.checkpoint_dir):
            raise NotADirectoryError(f"Checkpoint directory not found: {args.checkpoint_dir}")

        logging.info(f"Searching for checkpoints in {args.checkpoint_dir}")
        checkpoint_paths = utils.find_checkpoints(
            args.checkpoint_dir,
            pattern=args.pattern,
            last_n=args.num_checkpoints
        )

        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No checkpoints found matching pattern '{args.pattern}' in {args.checkpoint_dir}"
            )

        logging.info(f"Found {len(checkpoint_paths)} checkpoints matching pattern")

    else:
        raise ValueError("Must provide either --checkpoint-dir or --checkpoints")

    # Average the checkpoints
    avg_state = utils.average_checkpoints(checkpoint_paths)

    # Save averaged checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(avg_state, args.output)
    logging.info(f"Saved averaged checkpoint to {args.output}")

    # Print summary
    logging.info("\nSummary:")
    logging.info(f"  Number of checkpoints averaged: {len(checkpoint_paths)}")
    logging.info(f"  Output file: {args.output}")
    logging.info(f"  Output file size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    args = get_args()
    main(args)
