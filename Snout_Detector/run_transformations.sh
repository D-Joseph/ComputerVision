#!/bin/bash

# Prevent the Mac from sleeping until the script finishes
caffeinate -s &

# Run training with no augmentation
#echo "Running training with no augmentation..."
#python3 train.py 32

# Run training with flip augmentation
echo "Running training with flip augmentation..."
python3 train.py -t f -b 32

# Run training with rotation augmentation
echo "Running training with rotation augmentation..."
python3 train.py -t r -b 32

echo "Running training with flip and rotation augmentation..."
python3 train.py -t fr -b 32
echo "All training runs completed."
