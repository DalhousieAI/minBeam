# minimal beam search implementation compatible with minGPT

import torch
import torch.nn as nn
from einops import repeat, rearrange, reduce


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
        # print(batch_idx.size(), beam_idx.size(), idx.size())
        idx = rearrange(idx, "(b beams) t -> b beams t", beams=beams)[
            batch_idx.reshape(-1), beam_idx.reshape(-1)
        ]
        # print(idx.size())
        # idx = rearrange(idx, 'b beams t -> (b beams) t', beams=beams, t=t)
        idx_next = rearrange(idx_next, "b beams -> (b beams) ()", beams=beam_size)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)
        beams = beam_size
    idx = rearrange(idx, "(b beams) t -> b beams t", beams=beam_size)
    return idx, logp


# reference algorithm for Stochastic Beam Search https://arxiv.org/abs/1903.06059
# included in case I have time to code it
# \begin{algorithm}[H]
#   \centering
#   \scriptsize
#   \caption{StochasticBeamSearch($p_{\bm{\theta}}$, $k$)} \label{alg:stochastic_beam_search}
#   \begin{algorithmic}[1]
#   	  \STATE {\bfseries Input:} one-step probability distribution $p_{\bm{\theta}}$, beam/sample size $k$
#   	  \STATE \textnormal{Initialize } \textsc{beam} empty
#   	  \STATE add $(\bm{y}^N=\emptyset, \phi_N = 0, G_{\phi_N} = 0)$ to \textsc{beam}
#   	  \FOR{$t = 1, \ldots, \text{steps}$}
#   	    \STATE \textnormal{Initialize } \textsc{expansions} \textnormal{ empty}
#   	    \FOR{$(\bm{y}^S, \phi_{S}, G_{\phi_{S}}) \in \textsc{beam}$}
#   	        \STATE $Z \gets - \infty$
#   	        \FOR {$S' \in \text{Children}(S)$}
#   	            \STATE $\phi_{S'} \gets \phi_{S} + \log p_{\bm{\theta}}(\bm{y}^{S'} | \bm{y}^S)$
#   	            \STATE $G_{\phi_{S'}} \sim \text{Gumbel}(\phi_{S'})$
#   	            \STATE $Z \gets \max(Z, G_{\phi_{S'}})$
#   	        \ENDFOR
#   	        \FOR{$S' \in \text{Children}(S)$}
#   	            \STATE $\tilde{G}_{\phi_{S'}} \gets - \log( \exp( - G_{\phi_S} ) - \exp ( - Z) + \exp ( - G_{\phi_{S'}}) )$
#   	            \STATE add $(\bm{y}^{S'}, \phi_{S'}, \tilde{G}_{\phi_{S'}})$ to \textsc{expansions}
#   	        \ENDFOR
#   	    \ENDFOR
#   	    \STATE \textsc{beam} ${} \gets \textnormal{take } \text{top $k$}$ of \textsc{expansions} according to $\tilde{G}$
#   	  \ENDFOR
#   	  \STATE Return \textsc{beam}
#   \end{algorithmic}
# \end{algorithm}


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
    assert torch.all(torch.eq(_a[0], a[0]))
    assert torch.all(torch.eq(_b[0], b[0]))
    err = torch.abs(_a[1] - a[1]).max()
    assert torch.allclose(_a[1], a[1]), (err, _a[1].size(), a[1].size())
    err = torch.abs(_b[1] - b[1]).max()
    assert torch.allclose(_b[1], b[1]), (err, _b[1].size(), b[1].size())
    model = DummyModel(4, 10)
    idx = torch.tensor([[0, 1, 2, 3], [0, 2, 1, 3]])
    x, logp = beam(idx, model, max_new_tokens=3, beam_size=1)
    _x, _logp = beam(idx, model, max_new_tokens=3, beam_size=5)
    # we should find better states (or as good) with more beams, never worse
    assert _logp.max() >= logp.max()
    import timeit

    model = DummyModel(4, 10)
    N = 100
    inefficient_time = timeit.timeit(
        "inefficient_beam(idx, model, max_new_tokens=3, beam_size=1)",
        globals=globals(),
        number=N,
    )
    efficient_time = timeit.timeit(
        "beam(idx, model, max_new_tokens=3, beam_size=1)", globals=globals(), number=N
    )
    print("inefficient beam: ", inefficient_time / N)
    print("efficient beam:   ", efficient_time / N)
    assert inefficient_time > efficient_time
    # this improvement should get better for models that benefit from batched calls
