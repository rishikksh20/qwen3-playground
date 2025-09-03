import torch
import torch.nn.functional as F


def greedy_decoding(model, token_ids, max_new_tokens, eos_token_id=None):

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if (eos_token_id is not None
                    and torch.all(next_token == eos_token_id)):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0, recent_tokens=None):
    """
    Apply temperature, top-k, top-p, and repetition penalty sampling.
    """

    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Repetition penalty (penalize tokens that appeared recently)
    if repetition_penalty != 1.0 and recent_tokens is not None:
        for token in set(recent_tokens.tolist()):
            logits[..., token] /= repetition_penalty

    # Top-k filtering
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        mask = logits < values[..., -1, None]
        logits = logits.masked_fill(mask, float("-inf"))

    # Top-p (nucleus) filtering
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Mask tokens where cumulative prob > top_p
        sorted_mask = cumulative_probs > top_p
        # shift mask so at least 1 token is kept
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        # apply mask
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Sample
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

def advance_decoding(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    temperature=1.0,
    top_k=None,
    top_p=None,
    repetition_penalty=1.0,
    window_size=50,
):
    """
    Streaming text generation with advanced sampling.
    """
    model.eval()
    recent_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1, :]   # logits of last token
            next_token = sample_next_token(
                out,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                recent_tokens=token_ids[:, -window_size:].flatten()
            )

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            # Update sequence
            token_ids = torch.cat([token_ids, next_token], dim=1)
            recent_tokens.append(next_token.item())