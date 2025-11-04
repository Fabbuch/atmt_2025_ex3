import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel

def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    """Decodes a sequence without teacher forcing. Works by relying on the model's own predictions, rather than the ground truth (trg_)"""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for t in range(max_out_len):
        # Create target padding mask with correct batch dimension
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]
        # Ensure trg_pad_mask has shape (batch_size, seq_len)
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        # Forward pass: use only the generated tokens so far
        output = model(src_tokens, src_pad_mask, generated, trg_pad_mask).to(device)
        # Get the logits for the last time step
        next_token_logits = output[:, -1, :]  # last time step
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy

        # Append next token to each sequence
        generated = torch.cat([generated, next_tokens], dim=1)

        # Mark sequences as finished if EOS is generated
        finished = finished | (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break
    # Remove initial BOS token and anything after EOS
    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)
    return predicted_tokens


def decode_beam(
    model: Seq2SeqModel,
    src_tokens: torch.Tensor,
    src_pad_mask: torch.Tensor,
    max_out_len: int,
    tgt_tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    beam_size: int = 5,
    length_penalty: float = 0.6,
):
    """Single-sentence beam search decoding for Transformer models (no KV caching).

    Expects batch size == 1 for src tensors. Returns a list of token ids (without BOS, may include EOS).
    """
    assert src_tokens.size(0) == 1, "decode_beam expects batch size 1"
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    # beams: list of (sequence[t], cum_logprob)
    beams = [(torch.tensor([[BOS]], device=device, dtype=torch.long), 0.0)]
    finished = []  # list of (sequence, score)

    for _ in range(max_out_len):
        new_beams = []
        for seq, logp in beams:
            # If already finished (ended with EOS), keep it
            if seq[0, -1].item() == EOS:
                finished.append((seq.clone(), logp))
                continue

            # Build target padding mask
            trg_pad_mask = (seq == PAD).unsqueeze(1).unsqueeze(2)

            # Forward pass
            logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask).to(device)
            next_token_logits = logits[:, -1, :]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)

            # Top-k
            topk_logp, topk_idx = torch.topk(log_probs, k=beam_size, dim=-1)
            topk_logp = topk_logp.squeeze(0)
            topk_idx = topk_idx.squeeze(0)

            for i in range(beam_size):
                token = topk_idx[i].view(1, 1)
                token_logp = topk_logp[i].item()
                new_seq = torch.cat([seq, token.to(device)], dim=1)
                new_logp = logp + token_logp
                new_beams.append((new_seq, new_logp))

        # If we generated only finished sequences
        if not new_beams and finished:
            break

        # Keep top beam_size candidates
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

        # Early stop if all beams ended with EOS
        if all(b[0][0, -1].item() == EOS for b in beams):
            finished.extend(beams)
            break

    # If nothing finished, fall back to current beams
    if not finished:
        finished = beams

    # Apply length penalty: score = logp / ((5+len)^alpha / (6^alpha))
    def lp(seq_len):
        return ((5.0 + seq_len) ** length_penalty) / ((5.0 + 1.0) ** length_penalty)

    scored = []
    for seq, logp in finished:
        # remove initial BOS for length
        seq_wo_bos = seq[:, 1:]
        seq_len = seq_wo_bos.size(1)
        scored.append((seq, logp / lp(seq_len)))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_seq = scored[0][0]

    # Convert to plain list (drop leading BOS, keep EOS if present)
    out = best_seq[0].tolist()[1:]
    return out
