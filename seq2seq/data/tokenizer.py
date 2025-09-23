import os
import torch
import sentencepiece as spm


class BPETokenizer:
    """Wrapper class for a Byte-Pair Encoding (BPE) tokenizer using SentencePiece.\\
    **Only used for training** the tokenizer, inference uses the SentencePieceProcessor directly."""
    def __init__(self, language: str, vocab_size: int,
                     eos='</s>', bos='<s>', pad='<pad>', unk='<unk>'):
        self.LANG = language
        self.vocab_size = vocab_size
        self.tokenizer = None

        # Initialize special tokens
        self.eos = eos
        self.bos = bos
        self.pad = pad
        self.unk = unk


    def __len__(self):
        
        return self.tokenizer.GetPieceSize() if self.tokenizer else 0
    
    def __getitem__(self, idx: int):
        if self.tokenizer:
            # If idx is out of range, return the unknown token
            return self.tokenizer.id_to_piece(idx) if idx < self.tokenizer.GetPieceSize() else self.unk
        else:
            raise ValueError("Tokenizer is not yet trained or loaded.")
        
    def index(self, token):
        if self.tokenizer:
            # If token is not found, return the index of the unknown token
            return self.tokenizer.piece_to_id(token) if token in self.tokenizer else self.tokenizer.piece_to_id(self.unk)
        else:
            raise ValueError("Tokenizer is not yet trained or loaded.")

    def train_tokenizer(self, training_data: str, model_dir: str):
        """Trains the SentencePiece tokenizer on the parameters provided in the initialization of the class.\\
            The model is saved to the specified directory."""
        model_name = f'{self.LANG}-bpe-{self.vocab_size}'
        spm.SentencePieceTrainer.train(f'--input={training_data} --model_prefix={model_name} --pad_id=3 --vocab_size={self.vocab_size} --model_type=bpe --unk_piece={self.unk} --bos_piece={self.bos} --eos_piece={self.eos} --pad_piece={self.pad}')
        
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(f'{model_name}.model')

        assert self.tokenizer.get_piece_size() == self.vocab_size, "Vocabulary size does not match the expected size."

        # os.makedirs(model_dir, exist_ok=True)
        # if it exists, overwrite existing model in the save-directory
        if os.path.exists(os.path.join(model_dir, f'{model_name}.model')):
            os.remove(os.path.join(model_dir, f'{model_name}.model'))
        os.renames(f'{model_name}.model', os.path.join(model_dir, f'{model_name}.model'))
        os.remove(f'{model_name}.vocab')

    def load(self, model_path: str):
        """Loads the SentencePiece tokenizer from the specified directory."""
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(os.path.abspath(model_path))
        self.vocab_size = self.tokenizer.get_piece_size()

    @classmethod
    def load_from_model_only(cls, model_path: str, language: str):
        """Loads the SentencePiece tokenizer from the specified directory."""
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(model_path)
        vocab_size = tokenizer.get_piece_size()
        instance = cls(language=language, vocab_size=vocab_size)
        instance.tokenizer = tokenizer
        return instance

    def save_vocab(self, model_dir):
        """Saves vocabulary to a readable text-file.\\
        **NOTE**: this is only for inspection by the user, the Dictionary class uses the model files to load the actual vocabulary."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not yet trained.")
        vocab_file = os.path.join(os.path.normpath(model_dir), f'{self.LANG}-bpe-{self.vocab_size}.vocab')
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for id in range(self.tokenizer.get_piece_size()):
                piece = self.tokenizer.id_to_piece(id)
                # score = self.tokenizer.get_score(id)
                f.write(f"{piece} {id}\n")
        print(f"Vocabulary saved to {vocab_file}")

    def get_vocab_list(self):
        """Returns the vocabulary as a list of tokens."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not yet trained.")
        return [self.tokenizer.id_to_piece(id) for id in range(self.tokenizer.get_piece_size())]

    def encode_to_tensor(self, string, append_eos=True, append_bos=True, emit_unk=False, consumer=None):
        """Takes a string and encodes it into a tensor of token IDs.\\
            Includes an optional consumer callback for processing each token ID."""
        ids = self.tokenizer.Encode(string, out_type=int, add_eos=append_eos, emit_unk_piece=emit_unk)
        if consumer:
            for id in ids:
                consumer(id)
        return torch.IntTensor(ids)
