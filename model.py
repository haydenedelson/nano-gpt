import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation):
    activation_lwr = activation.lower()
    if activation_lwr == 'relu':
        return nn.ReLU()
    elif activation_lwr == 'gelu':
        return nn.GELU()
    else:
        raise AssertionError(f"{activation} activation not supported")


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=False, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        batch_size, seq_length, n_embed = x.shape

        q, k, v = self.qkv_proj(x).split(self.embed_dim, dim=2)
        # Transpose to (batch_size, num_heads, seq_length, head_size)
        q = q.view(batch_size, seq_length, self.num_heads, n_embed // self.num_heads).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, n_embed // self.num_heads).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, n_embed // self.num_heads).transpose(1, 2)

        att = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0, is_causal=True)
        # Re-assemble all head outputs side-by-side
        att = att.transpose(1, 2).contiguous().view(batch_size, seq_length, n_embed)

        out = self.out_proj(att)
        if self.dropout:
            out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, input_size, activation, expansion_factor=4, dropout=0.0):
        super(FeedForward, self).__init__()
        self.lin1 = nn.Linear(input_size, expansion_factor * input_size)
        self.activation = get_activation(activation)
        self.lin2 = nn.Linear(expansion_factor * input_size, input_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_bias, linear_expansion_factor, activation, dropout):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.sa = MultiHeadAttention(embed_dim, num_heads, bias=attn_bias, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffwd = FeedForward(embed_dim, activation, expansion_factor=linear_expansion_factor, dropout=dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        
        return x
    

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, num_layers, embed_dim, num_heads, attn_bias, linear_expansion_factor, activation, dropout):
        super(GPT, self).__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, embed_dim),
            wpe = nn.Embedding(block_size, embed_dim),
            dropout = nn.Dropout(dropout),
            hidden = nn.ModuleList([TransformerBlock(embed_dim, num_heads, attn_bias, linear_expansion_factor, activation, dropout) for _ in range(num_layers)]),
            ln_f = nn.LayerNorm(embed_dim)
        ))
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Tie LM head weights & token embedding weights
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight
    
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.dropout(tok_emb + pos_emb)
        for block in self.transformer.hidden:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

 