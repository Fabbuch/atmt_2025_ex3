#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=2:0:0
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=toy_rope_beam.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

echo "========================================="
echo "TOY RoPE + BEAM SEARCH + CHECKPOINT AVG"
echo "========================================="

# TRAIN RoPE with epoch checkpoints
python train.py \
    --cuda \
    --data toy_example/data/prepared/ \
    --src-tokenizer toy_example/tokenizers/cz-bpe-2000.model \
    --tgt-tokenizer toy_example/tokenizers/en-bpe-2000.model \
    --source-lang cz \
    --target-lang en \
    --batch-size 32 \
    --arch transformer_rope \
    --max-epoch 10 \
    --log-file toy_example/logs/train_rope.log \
    --save-dir toy_example/checkpoints_rope/ \
    --epoch-checkpoints \
    --save-interval 1 \
    --ignore-checkpoints \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 100 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3

echo ""
echo "✓ RoPE Training completed!"

# CHECKPOINT AVERAGING
echo "Averaging last 5 checkpoints..."
python average_checkpoints.py \
    --checkpoint-dir toy_example/checkpoints_rope/ \
    --num-checkpoints 5 \
    --output toy_example/checkpoints_rope/checkpoint_averaged.pt

echo "✓ Checkpoint averaging completed!"

# BEAM SEARCH with BEST checkpoint
echo ""
echo "Running Beam Search with BEST checkpoint..."
python translate.py \
    --cuda \
    --input toy_example/data/raw/test.cz \
    --src-tokenizer toy_example/tokenizers/cz-bpe-2000.model \
    --tgt-tokenizer toy_example/tokenizers/en-bpe-2000.model \
    --checkpoint-path toy_example/checkpoints_rope/checkpoint_best.pt \
    --beam-size 5 \
    --batch-size 1 \
    --max-len 100 \
    --output toy_example/results/rope_beam5_best.en \
    --bleu \
    --reference toy_example/data/raw/test.en

# BEAM SEARCH with AVERAGED checkpoint
echo ""
echo "Running Beam Search with AVERAGED checkpoint..."
python translate.py \
    --cuda \
    --input toy_example/data/raw/test.cz \
    --src-tokenizer toy_example/tokenizers/cz-bpe-2000.model \
    --tgt-tokenizer toy_example/tokenizers/en-bpe-2000.model \
    --checkpoint-path toy_example/checkpoints_rope/checkpoint_averaged.pt \
    --beam-size 5 \
    --batch-size 1 \
    --max-len 100 \
    --output toy_example/results/rope_beam5_averaged.en \
    --bleu \
    --reference toy_example/data/raw/test.en

echo ""
echo "========================================="
echo "ALL DONE! Results in toy_example/results/"
echo "========================================="
