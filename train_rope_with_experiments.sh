#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=rope_experiments.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

echo "========================================="
echo "RoPE Training with Epoch Checkpoints"
echo "========================================="

# PREPARE DATA
python preprocess.py \
    --source-lang cz \
    --target-lang en \
    --raw-data ~/shares/cz-en/data/raw \
    --dest-dir ./cz-en/data/prepared \
    --model-dir ./cz-en/tokenizers \
    --test-prefix test \
    --train-prefix train \
    --valid-prefix valid \
    --src-vocab-size 8000 \
    --tgt-vocab-size 8000 \
    --src-model ./cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-model ./cz-en/tokenizers/en-bpe-8000.model

# TRAIN with EPOCH CHECKPOINTS enabled
echo ""
echo "Starting RoPE training with epoch checkpoints..."
python train.py \
    --cuda \
    --data cz-en/data/prepared/ \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --source-lang cz \
    --target-lang en \
    --batch-size 64 \
    --arch transformer_rope \
    --max-epoch 7 \
    --log-file cz-en/logs/train_rope.log \
    --save-dir cz-en/checkpoints/ \
    --epoch-checkpoints \
    --save-interval 1 \
    --ignore-checkpoints \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 300 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3

echo ""
echo "✓ Training completed!"
echo ""

# CHECKPOINT AVERAGING
echo "========================================="
echo "Checkpoint Averaging"
echo "========================================="
python average_checkpoints.py \
    --checkpoint-dir cz-en/checkpoints/ \
    --num-checkpoints 5 \
    --output cz-en/checkpoints/checkpoint_averaged_last5.pt

echo "✓ Checkpoint averaging completed!"
echo ""

# TRANSLATION EXPERIMENTS
echo "========================================="
echo "Running Translation Experiments"
echo "========================================="

# Experiment 1: Greedy + Best Checkpoint
echo "Experiment 1: Greedy decoding with BEST checkpoint"
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --batch-size 64 \
    --max-len 300 \
    --output cz-en/results/translation_greedy_best.en \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en

# Experiment 2: Greedy + Averaged Checkpoint
echo ""
echo "Experiment 2: Greedy decoding with AVERAGED checkpoint"
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_averaged_last5.pt \
    --batch-size 64 \
    --max-len 300 \
    --output cz-en/results/translation_greedy_averaged.en \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en

# Experiment 3: Beam Search + Best Checkpoint
echo ""
echo "Experiment 3: Beam Search (beam=5) with BEST checkpoint"
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --beam-size 5 \
    --length-penalty 0.6 \
    --batch-size 1 \
    --max-len 300 \
    --output cz-en/results/translation_beam5_best.en \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en

# Experiment 4: Beam Search + Averaged Checkpoint
echo ""
echo "Experiment 4: Beam Search (beam=5) with AVERAGED checkpoint"
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_averaged_last5.pt \
    --beam-size 5 \
    --length-penalty 0.6 \
    --batch-size 1 \
    --max-len 300 \
    --output cz-en/results/translation_beam5_averaged.en \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en

echo ""
echo "========================================="
echo "All experiments completed!"
echo "Results saved in: cz-en/results/"
echo "========================================="
