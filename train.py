import torch
import torch.nn as nn
from torch.nn import functional as F

# test hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32
n_heads = 4
n_blocks = 4
dropout = 0.2

# hyperparameters
# batch_size = 64
# block_size = 256
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = "cuda" if torch.cuda.is_available() else "cpu"
# eval_iters = 200
# n_embd = 384
# n_heads = 6
# n_blocks = 6
# dropout = 0.2
# -------------

torch.manual_seed(1337)

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda es: "".join([itos[i] for i in es])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    idxs = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in idxs])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idxs])

    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

# class Head(nn.Module):
    # def __init__(self, head_size):
    #     super().__init__()
    #     self.head_size = head_size
    #     self.key = nn.Linear(n_embd, head_size, bias=False)
    #     self.query = nn.Linear(n_embd, head_size, bias=False)
    #     self.value = nn.Linear(n_embd, head_size, bias=False)
    #     self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    #     self.dropout = nn.Dropout(dropout)

    # def forward(self, x):
    #     B, T, C = x.shape
    #     k = self.key(x)  # (B,T,C), where C is head_size
    #     q = self.query(x)  # (B,T,C)

    #     wei = (
    #         q @ k.transpose(-2, -1) * (self.head_size**-0.5)
    #     )  # (B,T,C) @ (B, C, T) --> (B,T,T)
    #     wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
    #     wei = F.softmax(wei, dim=-1)  # (B,T,T)
    #     wei = self.dropout(wei)

    #     v = self.value(x)  # (B,T,C)
    #     out = wei @ v  # (B,T,T) @ (B,T,C) --> (B,T,C)
    #     return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd):
        super().__init__()
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.head_size = n_embd // num_heads

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B,T,3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # each is (B,T,C)
        
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # (B,n_heads,T,head_size)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # (B,n_heads,T,head_size)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # (B,n_heads,T,head_size)
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=dropout if self.training else 0.0, 
            is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out)  # (B,T,n_embd)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.sa_heads = MultiHeadAttention(num_heads, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, num_heads=n_heads) for _ in range(n_blocks)],
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of ints
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T, C)

        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
print(context.shape)
