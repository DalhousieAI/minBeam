# minimal beam search implementation compatible with minGPT

import torch
import torch.nn as nn
from einops import repeat, rearrange


@torch.no_grad()
def greedy(idx, model, max_new_tokens, drop_tokens=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    To remove some indexes from the sequence, you can specify a list of indexes to drop as
    drop_tokens.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = (
            idx if idx.size(1) <= model.block_size else idx[:, -model.block_size :]
        )
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # select the highest indexes from the final logits
        logits = logits[:, -1, :]
        if drop_tokens is not None:
            logits[:, drop_tokens] = -1e20
        _, idx_next = torch.topk(logits, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


@torch.no_grad()
def inefficient_beam(idx, model, max_new_tokens, beam_size, drop_tokens=None):
    "Beam search with inefficient nested loops to test against"
    b, t = idx.size()
    _idx = []
    _logprobs = []
    for x in idx:  # iterate over batch index
        x = x.view(1, t)
        # the first time we need to initialize the beams
        # if the sequence context is growing too long we must crop it at block_size
        x_cond = x if x.size(1) <= model.block_size else x[:, -model.block_size :]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(x_cond)
        # select the highest indexes from the final logits
        logcond = logits[:, -1, :]
        if drop_tokens is not None:
            logcond[:, drop_tokens] = -1e20
        logcond, idx_next = torch.topk(logcond, k=beam_size, dim=-1)
        # expand to beam_size
        x = repeat(x, "() t -> beam_size t", beam_size=beam_size)
        x = torch.cat((x, idx_next.view(-1, 1)), dim=1)
        logprobs = logcond.view(-1, 1)
        for _ in range(max_new_tokens - 1):
            # if the sequence context is growing too long we must crop it at block_size
            x_cond = x if x.size(1) <= model.block_size else x[:, -model.block_size :]
            # forward the model to get the logits for the index in the sequence
            logits, _ = model(x_cond)
            # select the highest indexes from the final logits
            logcond = logits[:, -1, :]
            if drop_tokens is not None:
                logcond[:, drop_tokens] = -1e20
            # find highest logprobs over all beams
            logprobs = logcond + logprobs
            most_probable_beams = []
            search_set = [
                (i, j) for i in range(beam_size) for j in range(model.vocab_size)
            ]
            while len(most_probable_beams) < beam_size:
                max_logprob = -1e20
                for i, j in search_set:
                    if logprobs[i, j] > max_logprob:
                        max_logprob = logprobs[i, j]
                        max_beam = i
                        max_token = j
                most_probable_beams.append((max_beam, max_token))
                search_set.remove((max_beam, max_token))
            # update the sequence with the most probable beams
            i = torch.tensor([i for i, j in most_probable_beams], dtype=torch.long)
            j = torch.tensor([j for i, j in most_probable_beams], dtype=torch.long)
            x = x[i, :]
            x = torch.cat((x, j.view(-1, 1)), dim=1)
            logprobs = logprobs[i, j].view(-1, 1)
        _idx.append(x)
        _logprobs.append(logprobs)
    return torch.stack(_idx), torch.stack(_logprobs).view(b, beam_size)


@torch.no_grad()
def beam(idx, model, max_new_tokens, beam_size, drop_tokens=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times by using beam search.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    To remove some indexes from the sequence, you can specify a list of indexes to drop as
    drop_tokens.
    """
    beams = 1
    b = idx.size(0)
    for new_token_idx in range(max_new_tokens):
        t = idx.size(1)
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = (
            idx if idx.size(1) <= model.block_size else idx[..., -model.block_size :]
        )
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        logp_cond = logits[..., -1, :]
        if drop_tokens is not None:
            logp_cond[:, drop_tokens] = -1e20
        # compute the beam log probabilities
        logp_cond = rearrange(
            logp_cond, "(b beams) v -> b beams v", beams=beams, v=model.vocab_size
        )
        if new_token_idx == 0:
            logp = logp_cond
        else:
            logp = logp_cond + logp.view(b, beams, 1)
        logp = rearrange(
            logp, "b beams v -> b (beams v)", beams=beams, v=model.vocab_size
        )
        # select the most probable beams
        logp, beam_vocab_idx = torch.topk(logp, k=beam_size, dim=-1)
        beam_idx = torch.div(beam_vocab_idx, model.vocab_size, rounding_mode="floor")
        idx_next = beam_vocab_idx % model.vocab_size  # blame copilot
        # this batch index selects beam_size elements from each batch
        batch_idx = repeat(
            torch.arange(b, device=idx.device), "b -> b beams", beams=beam_size
        )
        idx = rearrange(idx, "(b beams) t -> b beams t", beams=beams)[
            batch_idx.reshape(-1), beam_idx.reshape(-1)
        ]
        idx_next = rearrange(idx_next, "b beams -> (b beams) ()", beams=beam_size)
        # append indexes to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)
        beams = beam_size
    idx = rearrange(idx, "(b beams) t -> b beams t", beams=beam_size)
    return idx, logp


@torch.no_grad()
def shorter_beam(idx, model, max_new_tokens, beams, drop_tokens=None):
    "Shorter implementation of beam search (I think it's more memory intensive)"
    b = idx.size(0)
    for new_token_idx in range(max_new_tokens):
        t = idx.size(1)
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = (
            idx if idx.size(1) <= model.block_size else idx[..., -model.block_size :]
        )
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        logp_cond = logits[..., [-1], :]
        if drop_tokens is not None:
            logp_cond[:, drop_tokens] = -1e20
        logp = (
            logp_cond.view(b, beams, -1) + logp.view(b, beams, 1)
            if new_token_idx > 0
            else logp_cond
        )
        # enumerate all candidate beams
        idx_next = torch.arange(model.vocab_size, device=idx.device)
        _beams = beams if new_token_idx > 0 else 1
        all_candidates = torch.cat(
            [
                repeat(
                    idx,
                    "(b beams) t -> b (beams v) t",
                    beams=_beams,
                    v=model.vocab_size,
                    t=t,
                ),
                repeat(
                    idx_next,
                    "v -> b (beams v) ()",
                    v=model.vocab_size,
                    b=b,
                    beams=_beams,
                ),
            ],
            dim=2,
        )
        # select the most probable beams
        logp, beam_idxs = torch.topk(
            rearrange(
                logp, "b beams v -> b (beams v)", beams=_beams, v=model.vocab_size
            ),
            k=beams,
            dim=-1,
        )
        idx = torch.cat([x[i] for i, x in zip(beam_idxs, all_candidates)], dim=0)
    return idx.view(b, beams, -1), logp


class DummyModel(nn.Module):
    def __init__(self, block_size, vocab_size, delay=0.0):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.outputs = torch.randn(
            (self.vocab_size**self.block_size, self.vocab_size)
        )

    def forward(self, idx):
        b, t = idx.size()
        idx = idx * (self.vocab_size ** torch.arange(t).view(1, t))
        return self.outputs[idx.view(-1)].view(b, t, self.vocab_size), None


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DummyModel(4, 3)
    idx = torch.tensor([[0, 1, 2], [0, 2, 1]])
    # print(model(idx))
    print(greedy(idx, model, max_new_tokens=3))
    a = inefficient_beam(idx, model, max_new_tokens=3, beam_size=1)
    print(a)
    b = inefficient_beam(idx, model, max_new_tokens=3, beam_size=2)
    print(b)
    _a = beam(idx, model, max_new_tokens=3, beam_size=1)
    print(_a)  # should match greedy with one beam
    _b = beam(idx, model, max_new_tokens=3, beam_size=2)
    print(_b)
    sa = shorter_beam(idx, model, max_new_tokens=3, beams=1)
    sb = shorter_beam(idx, model, max_new_tokens=3, beams=2)
    print(sb)
    assert torch.all(torch.eq(_a[0], a[0]))
    assert torch.all(torch.eq(_b[0], b[0]))
    assert torch.all(torch.eq(a[0], sa[0]))
    err = torch.abs(_a[1] - a[1]).max()
    assert torch.allclose(_a[1], a[1]), (err, _a[1].size(), a[1].size())
    err = torch.abs(_b[1] - b[1]).max()
    assert torch.allclose(_b[1], b[1]), (err, _b[1].size(), b[1].size())
    err = torch.abs(sa[1] - _a[1]).max()
    assert torch.allclose(sa[1], _a[1]), (err, sa[1].size(), _a[1].size())
    model = DummyModel(4, 10)
    idx = torch.tensor([[0, 1, 2, 3], [0, 2, 1, 3]])
    x, logp = beam(idx, model, max_new_tokens=3, beam_size=1)
    _x, _logp = beam(idx, model, max_new_tokens=3, beam_size=5)
    # we should find better states (or as good) with more beams, never worse
    assert _logp.max() >= logp.max()
    import timeit

    model = DummyModel(4, 10)
    N = 32
    inefficient_time = timeit.timeit(
        "inefficient_beam(idx, model, max_new_tokens=10, beam_size=5)",
        globals=globals(),
        number=N,
    )
    efficient_time = timeit.timeit(
        "beam(idx, model, max_new_tokens=10, beam_size=5)", globals=globals(), number=N
    )
    shorter_time = timeit.timeit(
        "shorter_beam(idx, model, max_new_tokens=10, beams=5)",
        globals=globals(),
        number=N,
    )
    print("inefficient beam: ", inefficient_time / N)
    print("efficient beam:   ", efficient_time / N)
    print("shorter beam:     ", shorter_time / N)
    assert inefficient_time > efficient_time
    # this improvement should get better for models that benefit from batched calls
