#!/usr/bin/bash
# Training script for beam search and checkpoint averaging experiments
# This will save epoch checkpoints for averaging later

# Clean up old results (optional)
rm -rf toy_example/checkpoints_experiment
rm -rf toy_example/logs_experiment

# TRAIN with epoch checkpoints enabled
python train.py \
    --data toy_example/data/prepared/ \
    --src-tokenizer toy_example/tokenizers/cz-bpe-1000.model \
    --tgt-tokenizer toy_example/tokenizers/en-bpe-1000.model \
    --source-lang cz \
    --target-lang en \
    --batch-size 32 \
    --arch transformer \
    --max-epoch 10 \
    --log-file toy_example/logs_experiment/train.log \
    --save-dir toy_example/checkpoints_experiment/ \
    --ignore-checkpoints \
    --epoch-checkpoints \
    --save-interval 1 \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 100 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3

echo "✓ Training completed!"
echo "Checkpoints saved in: toy_example/checkpoints_experiment/"

# CHECKPOINT AVERAGING
echo ""
echo "Averaging last 5 checkpoints..."
python average_checkpoints.py \
    --checkpoint-dir toy_example/checkpoints_experiment/ \
    --num-checkpoints 5 \
    --output toy_example/checkpoints_experiment/checkpoint_averaged_last5.pt

echo "✓ Checkpoint averaging completed!"

# TRANSLATE with different methods
echo ""
echo "Running translations with different methods..."

# 1. Greedy decoding with best checkpoint
python translate.py \
    --input toy_example/data/raw/test.cz \
    --src-tokenizer toy_example/tokenizers/cz-bpe-1000.model \
    --tgt-tokenizer toy_example/tokenizers/en-bpe-1000.model \
    --checkpoint-path toy_example/checkpoints_experiment/checkpoint_best.pt \
    --batch-size 32 \
    --max-len 100 \
    --output toy_example/results/translation_greedy.en \
    --bleu \
    --reference toy_example/data/raw/test.en

# 2. Beam search (beam=5) with best checkpoint
python translate.py \
    --input toy_example/data/raw/test.cz \
    --src-tokenizer toy_example/tokenizers/cz-bpe-1000.model \
    --tgt-tokenizer toy_example/tokenizers/en-bpe-1000.model \
    --checkpoint-path toy_example/checkpoints_experiment/checkpoint_best.pt \
    --batch-size 1 \
    --max-len 100 \
    --beam-size 5 \
    --length-penalty 0.6 \
    --output toy_example/results/translation_beam5.en \
    --bleu \
    --reference toy_example/data/raw/test.en

# 3. Greedy with averaged checkpoint
python translate.py \
    --input toy_example/data/raw/test.cz \
    --src-tokenizer toy_example/tokenizers/cz-bpe-1000.model \
    --tgt-tokenizer toy_example/tokenizers/en-bpe-1000.model \
    --checkpoint-path toy_example/checkpoints_experiment/checkpoint_averaged_last5.pt \
    --batch-size 32 \
    --max-len 100 \
    --output toy_example/results/translation_averaged_greedy.en \
    --bleu \
    --reference toy_example/data/raw/test.en

# 4. Beam search with averaged checkpoint
python translate.py \
    --input toy_example/data/raw/test.cz \
    --src-tokenizer toy_example/tokenizers/cz-bpe-1000.model \
    --tgt-tokenizer toy_example/tokenizers/en-bpe-1000.model \
    --checkpoint-path toy_example/checkpoints_experiment/checkpoint_averaged_last5.pt \
    --batch-size 1 \
    --max-len 100 \
    --beam-size 5 \
    --length-penalty 0.6 \
    --output toy_example/results/translation_averaged_beam5.en \
    --bleu \
    --reference toy_example/data/raw/test.en

echo ""
echo "✓ All experiments completed!"
echo "Results saved in: toy_example/results/"
