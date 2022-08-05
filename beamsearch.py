# minimal beam search implementation compatible with minGPT

import torch
import torch.nn as nn
from einops import repeat, rearrange, reduce


# reference sampling function from minGPT
# @torch.no_grad()
# def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, drop_tokens=None):
#     """
#     Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
#     the sequence max_new_tokens times, feeding the predictions back into the model each time.
#     Most likely you'll want to make sure to be in model.eval() mode of operation for this.
#     """
#     for _ in range(max_new_tokens):
#         # if the sequence context is growing too long we must crop it at block_size
#         idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
#         # forward the model to get the logits for the index in the sequence
#         logits, _ = self(idx_cond)
#         # pluck the logits at the final step and scale by desired temperature
#         logits = logits[:, -1, :] / temperature
#         # if there are tokens to drop, subtract large negative from their logits
#         if drop_tokens is not None:
#             logits[:, drop_tokens] = -1e20
#         # optionally crop the logits to only the top k options
#         if top_k is not None:
#             v, _ = torch.topk(logits, top_k)
#             logits[logits < v[:, [-1]]] = -float('Inf')
#         # apply softmax to convert logits to (normalized) probabilities
#         probs = F.softmax(logits, dim=-1)
#         # either sample from the distribution or take the most likely element
#         if do_sample:
#             idx_next = torch.multinomial(probs, num_samples=1)
#         else:
#             _, idx_next = torch.topk(probs, k=1, dim=-1)
#         # append sampled index to the running sequence and continue
#         idx = torch.cat((idx, idx_next), dim=1)
#
#     return idx

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
        idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # select the highest indexes from the final logits
        logprobs = logits[:, -1, :]
        if drop_tokens is not None:
            logits[:, drop_tokens] = -1e20
        _, idx_next = torch.topk(logprobs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

@torch.no_grad()
def beam(idx, model, max_new_tokens, beam_size, drop_tokens=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times by using beam search.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    To remove some indexes from the sequence, you can specify a list of indexes to drop as
    drop_tokens.
    """
    b, t = idx.size()
    beams = 1
    beam_log_probs = torch.zeros((b, beam_size), device=idx.device)
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= model.block_size else idx[..., -model.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        logp_cond = logits[..., -1, :]
        if drop_tokens is not None:
            logits[:, drop_tokens] = -1e20
        # compute the beam log probabilities
        logp_cond = rearrange(logp_cond, '(b beams) v -> b beams v', beams=beams, v=model.vocab_size)
        if beams > 1:
            logp = logp_cond + beam_log_probs.view(b, beams, 1)
        else: # on first iteration no need to add previous beam log probs
            logp = logp_cond
        logp = rearrange(logp, 'b beams v -> b (beams v)', beams=beams, v=model.vocab_size)
        # select the most probable beams
        logp, beam_vocab_idx = torch.topk(logp, k=beam_size, dim=-1)
        beam_idx = torch.div(beam_vocab_idx, model.vocab_size, rounding_mode='floor')
        idx_next = beam_vocab_idx % model.vocab_size
        idx = rearrange(idx, '(b beams) t -> b beams t', beams=beams)[:, beam_idx]
        beams = beam_size
        print(idx.size(), beam_idx.size())
        assert False, "broken"
        idx = rearrange(idx, 'b beams t -> (b beams) t', beams=beams, t=t)
        idx_next = rearrange(idx_next, 'b beams -> (b beams) ()', beams=beam_size)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)
        # update the beam log probabilities
        for i, p, q in zip(idx, beam_log_probs.view(-1), logp.view(-1)):
            print(i, p, q, p+q)
        beam_log_probs += logp
        beams = beam_size
    idx = rearrange(idx, '(b beams) t -> b beams t', beams=beam_size)
    return idx, beam_log_probs

# reference algorithm for Stochastic Beam Search https://arxiv.org/abs/1903.06059
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
    def __init__(self):
        super().__init__()
        self.block_size = 4
        self.vocab_size = 3
        self.outputs = torch.randn((self.vocab_size**self.block_size, self.vocab_size))

    def forward(self, idx):
        b, t = idx.size()
        idx = idx*(self.vocab_size**torch.arange(t).view(1, t))
        return self.outputs[idx.view(-1)].view(b, t, self.vocab_size), None

if __name__ == "__main__":
    torch.manual_seed(0)
    model = DummyModel()
    idx = torch.tensor([[0, 1, 2], [0, 2, 1]])
    #print(model(idx))
    print(greedy(idx, model, max_new_tokens=3))
    print(beam(idx, model, max_new_tokens=3, beam_size=1)) # should match greedy with one beam
    print(beam(idx, model, max_new_tokens=3, beam_size=2))
