import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions? 257번째 맞추자 어
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # 신경망이 커질수록 학습률을 낮게!
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # head = 4 -> n_embd//n_head = 64 모든 헤드가 64차원임
n_head = 6      
n_layer = 6     
dropout = 0.2 # 계산의 20%가 비활성화됨.

# ------ load dataset ------
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#print(text[:1000])

# ------ tokenize ------
chars = sorted(list(set(text)))
vocab_size = len(chars)
#print(''.join(chars))
#print(vocab_size)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#print(encode("hii there"))
#print(decode(encode("hii there")))

# let's now encode the entire text dataset and store it into a torch.Tensor
import torch # we use PyTorch: https://pytorch.org
data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

# --------------

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data

    # split이 train인 경우에는 train_data를 살펴보고 image.png
    # split이 train이 그렇지 않다면 validation data을 살펴볼 것이다.

    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 0과 (데이터의 길이 - block_size(청크 사이즈)) 사이에 난수를
    # 4개 (batch_size) ix 에 생성해주는 코드 

    # 각 청크는 1-dimension chuck임, 이 예시에서는 [1, 8]

    # 근데 torch.stack 함수를 사용해서 1-dimension chuck들을 열 별로 쌓아
    # [4, 8] tensor로 만들어줌! (batch_size, block_size)

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x , y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,C)
        q = self.query(x) #(B,T,C)
        v = self.value(x) #(B,T,C)

        # computing attention scores ("affinities")
        weight = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        weight = F.softmax(weight, dim=-1) # (B,T,T)
        weight = self.dropout(weight)
        out = weight @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # proj : 여러 헤드의 지혜를 하나로 합치는 레이어 ,, num_head 만큼 수행!
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out) # proj : 여러 헤드의 지혜를 하나로 합치는 레이어
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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
    """ Transformer block : communication followed by computation """ 
    # 물론 cross-attention을 제외함.

    def __init__(self, n_embd, n_head):
        # n_embd : embedding dimension , n_head : the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head # 흥미롭지 않나요?
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) # residual ...
        return x

# Super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        """self.blocks = nn.Sequential(
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4),
            nn.LayerNorm(n_embd),
        )"""
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)  # output layer

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers

        # idx를 전달하게 되면 입력의 모든 정수가 this 임베딩 테이블을 참조
        # e.g. , 24가 input으로 들어가게 되면 24번째 행을 뽑음 
        token_embeddings = self.token_embedding_table(idx) #(B,T,C) C = n_embd
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = token_embeddings + position_embeddings #(B,T,C) 
        x = self.blocks(x) # (B, T, C)
        logits = self.lm_head(x) #(B,T,vocab_size) 
        # lm_head : decoder language modeling head
        # (B,T,C) batch, time , channel

        # batch = 4, time = 8, channel(vocab_size) = 65

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 시퀀스를 1차원 시퀀스로 쭉 늘려버리자! , 배열을 2차원으로 늘이는 것과 같다.
            targets = targets.view(B*T)
          # targets = targets.view(-1)

            loss = F.cross_entropy(logits, targets) # measure qaulity of logits

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop index to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C) => prediction
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) -> batch마다 단일 예측 
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
model = model.to(device)


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters): # increase number of steps for good results...

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
   
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
