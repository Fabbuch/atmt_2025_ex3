# ATMT Codebase

Materials for "Advanced Techniques of Machine Translation" (UZH, HS25).

**Please refer to the assignment sheet for instructions for the actual assignments.**

The toolkit is based on last year's [version](https://github.com/davidguzmanp/atmt_2024/), but with significant upgrades, such as BPE-tokenization (using [Sentencepiece](https://github.com/google/sentencepiece)) and the change from an RNN to a transformer architecture (based on [this implementation](https://github.com/AliAbdien/Transformer-based-Machine-Translation-from-Scratch))


## Environment Setup

The coding assignments are designed to be run on UZH's ScienceCluster, documentation for which can be found [here](https://docs.s3it.uzh.ch/cluster/overview/). As part of a pilot program, we've been given access to a certain amount of GPU-hours, which we can use to train more sophisticated models than in the last year's assignments. Note that the models trained on the cluster will likely not reach SOTA performance, but should be more interesting than just toy examples nevertheless.

### Connecting to the cluster

Follow the steps outlined in the [ScienceCluster documentation](https://docs.s3it.uzh.ch/cluster/overview/) to connect to the cluster.



#### Local dry-run

In case you'd like to try out the code locally, it is recommended to run `toy_example.sh` instead of the "real" training script. It uses a small fraction of the full dataset contained in the `toy_example` directory.

We strongly suggest creating a Python environment to prevent library clashes with future projects, using either Conda or virtualenv (Conda is suggested). For other options, see [supplementary material](https://neat-tortellini-10f.notion.site/ATMT-Autumn-2023-Assignment-1-Setup-Instructions-96d8444a7d7146139a5b76a86a559f5f?pvs=4)


### Installing required packages

The ScienceCluster uses Mamba (an improved version of Conda) to setup environments.

To install the required packages, run the following lines of code on the cluster:

#TODO
```
module load mamba

mamba create -n atmt -c pytorch -c nvidia pytorch pytorch-cuda
```
<!-- ### conda

```
# ensure that you have conda (or miniconda) installed (https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and that it is activated

# create clean environment
conda create --name atmt311 python=3.11

# activate the environment
conda activate atmt311

# intall required packages
conda install pytorch=2.0.1 numpy tqdm sacrebleu
```

### virtualenv

```
# ensure that you have python > 3.6 downloaded and installed (https://www.python.org/downloads/)

# install virtualenv
pip install virtualenv  # for both powershell and WSL

# create a virtual environment named "atmt311"
virtualenv --python=python3.11 atmt311  # on WSL terminal
python -m venv atmt311    # on powershell

# launch the newly created environment
source atmt311/bin/activate
.\atmt311\Scripts\Activate.ps1   # on powershell


# intall required packages
pip install torch==2.0.1 numpy tqdm sacrebleu   # for both powershell and WSL
``` -->

<!-- # Data Preprocessing

```
# normalise, tokenize and truecase data
bash scripts/extract_splits.sh ../infopankki_raw data/en-sv/infopankki/raw

# binarize data for model training
bash scripts/run_preprocessing.sh data/en-sv/infopankki/raw/
``` -->


## Usage Example

Let's have a look at what how to use the repository.

The following steps are a working example on the data in the directory `toy_example`.

<details>

<summary>Example File Structure:</summary>

```
toy_example/
└── data/
    └── raw/
        ├── train.en
        ├── train.de
        ├── valid.en
        ├── valid.de
        ├── test.en
        └── test.de
```

</details>

### pre-processing

Here, the following things happen:
- A separate **BPE-tokenizer** is trained for each side (source and target), with the specified vocab-size
- Each split will be tokenized using the (corresponding language's) tokenizer, turning each sentence from a string into a tensor containing **token-IDs**
- Each split is _pickled_ (stored in binary format) in the specified `--dest-dir`-folder
- 

```bash
python preprocess.py \
    --source-lang cz \  # the tag of the source language (Czech)
    --target-lang en \  # the tag of the target language (English)
    --raw-data .\toy_example\data\raw \  # unprocessed .txt-files, usually named <split>.<tag>, e.g. train.cz
    --dest-dir .\toy_example/data/prepared \  # where the processed (tokenized, pickled) files will be stored to
    --model-dir toy_example\tokenizers \  # where to store the trained tokenization models
    --test-prefix test \  # expected prefix for files belonging to the test-split
    --train-prefix train \  # expected prefix for files belonging to the train-split
    --valid-prefix valid \  # expected prefix for files belonging to the validation-split
    --src-vocab-size 1000 \  # size of the source (BPE-)vocab
    --tgt-vocab-size 1000 \  # size of the target (BPE-)vocab
    --ignore-existing \  # overwrite existing processed files (only recommended for testing)
    --force-train  # force training, even if a tokenizer (same language and vocab-size) already exists
```

Notes:
- The data to train the tokenizer is expected to be stored as raw text under the name `<train-prefix>.<tag>` in the specified `raw-data` directory
<br>e.g. `.\toy_example\data\raw\train.en`
- Filename of the resulting tokenizer: `<tag>-bpe-<vocab_size>.model`
<br>e.g. `cz-bpe-1000.model`
- The `.vocab` files exist just to have a user-readable version of the vocabulary. All actual (de-)tokenization will use the `.model` files

-   <details>

    <summary> Resulting file structure </summary> 

    ```
    toy_example/
    ├── data/
    │   ├── raw/
    │   │   ├── train.en
    │   │   ├── train.de
    │   │   ├── valid.en
    │   │   ├── valid.de
    │   │   ├── test.en
    │   │   └── test.de
    │   └── prepared/
    │       ├── train.en
    │       ├── train.de
    │       ├── valid.en
    │       ├── valid.de
    │       ├── test.en
    │       └── test.de
    └── tokenizers/
        ├── en-bpe-1200.model
        ├── de-bpe-1200.model
        ├── en-bpe-1200.vocab
        └── de-bpe-1200.vocab
    ```

    </details>

### Training a model

Trains a transformer model on the prepared data.

<details>

<summary> More details </summary>

1. Build model with the specified parameters & load data with the provided tokenizers.
2. During each epoch, the model iterates over the **training** data in batches, computes the loss using teacher forcing, performs backpropagation, applies gradient , and updates the model parameters with the optimizer.
3. At the end of each epoch,
    - calculate the **validation** loss and perplexity _with teacher forcing_
    - generate translations _without teacher forcing_ and compute BLEU from it
4. Training stops early if the validation loss does not improve for a set number of epochs (patience).
5. The model's performance is evaluated on a final, unseen **test set**

</details>

<br>

```bash
python train.py \
    --data toy_example/data/prepared/ \  # output of preprocess.py
    --src-tokenizer toy_example/tokenizers/cz-bpe-1000.model \  # location of the source tokenizer model
    --tgt-tokenizer toy_example/tokenizers/en-bpe-1000.model \  # location of the target tokenizer model
    --source-lang cz \
    --target-lang en \
    --batch-size 32 \  # batch size for training and validation
    --arch transformer \  # architecture variant: by default, only transformer exists
    --max-epoch 10 \  # maximum number of epochs (early stopping possible)
    --log-file toy_example/logs/train.log \  # log-file name and location
    --save-dir toy_example/checkpoints/ \  # directory to save the model checkpoints to
    --ignore-checkpoints \  # ignore potential existing checkpoints & train from scratch
    --encoder-dropout 0.1 \ 
    --decoder-dropout 0.1 \
    --dim-embedding 256 \  # embedding (and general model) dimension
    --attention-heads 4 \  # number of attention heads
    --dim-feedforward-encoder 1024 \  # dimension of Encoder-FFNs
    --dim-feedforward-decoder 1024 \  # dimension of Decoder-FFNs
    --max-seq-len 100 \  # maximum sequence length (longer inputs/outputs will be trimmed to the specified number of tokens)
    --n-encoder-layers 3 \  # number of encoder layers
    --n-decoder-layers 3  # number of decoder layers
```

Notes:
- add the `--cuda` flag if you want to train on a GPU, e.g. using Google Colab
- the source- and target-tokenizer arguments have to match the model files created in the pre-processing step
- the progress-bar is hardcoded to update only in 2-second intervals. This is done to not reduce the clutter in the output file of actual training runs on the cluster
- the model specific arguments are added to `args` in the python file implementing said model, in this case `seq2seq/models/transformer.py`

# Translation

Translate a raw (source language) text file (one sentence per line) to the target language
```bash
python translate.py \
    --input toy_example/data/raw/test.cz \  #  Path to the raw source text file (one sentence per line, in Czech)
    --src-tokenizer toy_example/tokenizers/cz-bpe-1000.model \  # Path to the trained SentencePiece tokenizer model for the source language
    --tgt-tokenizer toy_example/tokenizers/en-bpe-1000.model \  # Path to the trained SentencePiece tokenizer model for the target language
    --checkpoint-path toy_example/checkpoints/checkpoint_best.pt \ # Path to the trained model checkpoint
    --batch-size 1 \  # Number of sentences to process in each batch 
    --max-len 100 \  # Maximum length of generated translations (in tokens)
    --output toy_example/toy_example_output.en \  # Path to write the generated translations (one per line)
    --bleu \  # If set, compute BLEU score after translation (score output vs. reference)
    --reference toy_example/data/raw/test.en  # Path to the reference translation file (one sentence per line, in English)
```

Notes:
- The source and target language tags do not have to be passed, as they are loaded and parsed as part of the model checkpoint
- If `--bleu` is set but no reference is provided, this will throw an error


# Assignments

Assignments must be submitted on OLAT by 14:00 on their respective
due dates.

