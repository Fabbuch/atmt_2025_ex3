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


def decode_beam_batched(
    model: Seq2SeqModel,
    src_tokens: torch.Tensor,
    src_pad_mask: torch.Tensor,
    max_out_len: int,
    tgt_tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    beam_size: int = 5,
    length_penalty: float = 0.6,
):
    """Batched beam search decoding for Transformer models.

    Processes multiple sentences in parallel with beam search.
    Returns a list of lists of token ids (one list per input sentence).

    Args:
        model: The Seq2Seq model
        src_tokens: Source tokens tensor (batch_size, src_len)
        src_pad_mask: Source padding mask (batch_size, 1, 1, src_len)
        max_out_len: Maximum output length
        tgt_tokenizer: Target tokenizer
        device: Device to use
        beam_size: Number of beams per sentence
        length_penalty: Length penalty alpha for scoring

    Returns:
        List of decoded sequences (without BOS, may include EOS)
    """
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    # Expand source tokens and masks for beam search
    # Shape: (batch_size * beam_size, src_len)
    src_tokens_expanded = src_tokens.unsqueeze(1).repeat(1, beam_size, 1).view(batch_size * beam_size, -1)
    src_pad_mask_expanded = src_pad_mask.unsqueeze(1).repeat(1, beam_size, 1, 1, 1).view(
        batch_size * beam_size, 1, 1, -1
    )

    # Initialize beams: (batch_size * beam_size, 1) with BOS
    current_beams = torch.full((batch_size * beam_size, 1), BOS, dtype=torch.long, device=device)

    # Beam scores: (batch_size, beam_size)
    beam_scores = torch.zeros(batch_size, beam_size, device=device)
    beam_scores[:, 1:] = float('-inf')  # Only first beam is active initially

    # Track finished sentences: (batch_size, beam_size)
    finished = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

    for step in range(max_out_len):
        # Create target padding mask
        trg_pad_mask = (current_beams == PAD).unsqueeze(1).unsqueeze(2)

        # Forward pass: (batch_size * beam_size, seq_len, vocab_size)
        with torch.no_grad():
            logits = model(src_tokens_expanded, src_pad_mask_expanded, current_beams, trg_pad_mask)

        # Get logits for last position: (batch_size * beam_size, vocab_size)
        next_token_logits = logits[:, -1, :]
        vocab_size = next_token_logits.size(-1)

        # Log probabilities: (batch_size * beam_size, vocab_size)
        log_probs = torch.log_softmax(next_token_logits, dim=-1)

        # Reshape to (batch_size, beam_size, vocab_size)
        log_probs = log_probs.view(batch_size, beam_size, vocab_size)

        # Add current beam scores: (batch_size, beam_size, vocab_size)
        scores = log_probs + beam_scores.unsqueeze(-1)

        # For finished beams, only allow PAD token to maintain score
        scores[finished] = float('-inf')
        scores[finished, :, PAD] = beam_scores[finished]

        # Reshape to (batch_size, beam_size * vocab_size) for top-k selection
        scores = scores.view(batch_size, -1)

        # Get top beam_size candidates for each sentence
        topk_scores, topk_indices = torch.topk(scores, k=beam_size, dim=-1)

        # Calculate which beam and which token each top-k corresponds to
        prev_beam_ids = topk_indices // vocab_size  # (batch_size, beam_size)
        token_ids = topk_indices % vocab_size  # (batch_size, beam_size)

        # Gather previous beams
        # Shape: (batch_size, beam_size, seq_len)
        current_beams_reshaped = current_beams.view(batch_size, beam_size, -1)

        # Expand indices for gathering
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, beam_size)
        selected_beams = current_beams_reshaped[batch_indices, prev_beam_ids]  # (batch_size, beam_size, seq_len)

        # Append new tokens: (batch_size, beam_size, seq_len + 1)
        current_beams = torch.cat([selected_beams, token_ids.unsqueeze(-1)], dim=-1)

        # Update beam scores
        beam_scores = topk_scores

        # Update finished status
        finished = finished[batch_indices, prev_beam_ids] | (token_ids == EOS)

        # Flatten back to (batch_size * beam_size, seq_len)
        current_beams = current_beams.view(batch_size * beam_size, -1)

        # Early stopping: if all beams are finished for all sentences
        if finished.all():
            break

    # Apply length penalty and select best beam for each sentence
    current_beams = current_beams.view(batch_size, beam_size, -1)

    def lp(seq_len):
        return ((5.0 + seq_len) ** length_penalty) / ((5.0 + 1.0) ** length_penalty)

    # Calculate lengths (excluding BOS)
    lengths = (current_beams != PAD).sum(dim=-1).float() - 1  # (batch_size, beam_size)

    # Apply length penalty to scores
    normalized_scores = beam_scores / lp(lengths)

    # For finished beams that haven't generated EOS, penalize them
    # This shouldn't happen often, but handle it gracefully

    # Select best beam for each sentence
    best_beam_ids = normalized_scores.argmax(dim=-1)  # (batch_size,)

    # Gather best sequences
    batch_indices = torch.arange(batch_size, device=device)
    best_sequences = current_beams[batch_indices, best_beam_ids]  # (batch_size, seq_len)

    # Convert to list of lists, removing BOS and truncating at EOS
    results = []
    for seq in best_sequences:
        seq_list = seq.tolist()
        # Remove BOS (first token)
        if seq_list[0] == BOS:
            seq_list = seq_list[1:]
        # Truncate at EOS (keep EOS for compatibility with original function)
        if EOS in seq_list:
            eos_idx = seq_list.index(EOS)
            seq_list = seq_list[:eos_idx]
        # Remove PAD tokens
        seq_list = [token for token in seq_list if token != PAD]
        results.append(seq_list)

    return results
