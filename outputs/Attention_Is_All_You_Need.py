import subprocess, sys

packages = [
    "torch",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn"
]

with open("requirements_notebook.txt", "w") as f:
    f.write("\n".join(packages))

# Ensure pip is available (uv venvs don't bundle pip by default)
subprocess.run(
    [sys.executable, "-m", "ensurepip", "--upgrade"],
    capture_output=True
)

print("Installing dependencies...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements_notebook.txt", "-q"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("Install error:")
    print(result.stderr)
    raise SystemExit(1)
print("All dependencies installed!")

# ## 2. Imports
import math
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# ## 3. Configuration & Hyperparameters
class Config:
    N = 6  # Number of encoder/decoder layers
    d_model = 512  # Embedding dimension
    d_ff = 2048  # Feed forward dimension
    h = 8  # Number of attention heads
    d_k = 64  # Key dimension
    d_v = 64  # Value dimension
    dropout_rate = 0.1
    label_smoothing = 0.1
    warmup_steps = 4000
    beta1 = 0.9
    beta2 = 0.98
    epsilon = 1e-9
    beam_size = 4
    length_penalty_alpha = 0.6
    max_len = 5000  # Maximum sequence length for positional encodings
    batch_size = 32
    epochs = 10
    lr = 0.0001
    
config = Config()

# ## 4. Data Loading
def load_data():
    """
    Load or create synthetic data for demonstration purposes.
    In practice, you would load the WMT datasets here.
    """
    print("Loading data...")
    
    # Create synthetic data for demonstration
    english_sentences = [
        "The cat sat on the mat.",
        "Dogs are loyal animals.",
        "I enjoy reading books.",
        "Machine learning is fascinating.",
        "Natural language processing enables computers to understand text."
    ] * 100  # Repeat to simulate larger dataset
    
    german_sentences = [
        "Die Katze saß auf der Matte.",
        "Hunde sind treue Tiere.",
        "Ich lese gerne Bücher.",
        "Maschinelles Lernen ist faszinierend.",
        "Die natürliche Sprachverarbeitung ermöglicht es Computern, Text zu verstehen."
    ] * 100  # Repeat to simulate larger dataset
    
    return english_sentences, german_sentences

try:
    src_data, tgt_data = load_data()
    print(f"Loaded {len(src_data)} sentence pairs")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Creating dummy data instead...")
    src_data = ["Hello world"] * 1000
    tgt_data = ["Hallo Welt"] * 1000

# ## 5. Data Preprocessing
class Vocabulary:
    def __init__(self, sentences, tokenizer=None):
        self.tokenizer = tokenizer or str.split
        self.word2index = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.index2word = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.word_freq = Counter()
        self.build_vocab(sentences)
        
    def build_vocab(self, sentences):
        for sentence in sentences:
            tokens = self.tokenizer(sentence.lower())
            self.word_freq.update(tokens)
            
        # Add words to vocabulary
        for word in self.word_freq:
            if word not in self.word2index:
                idx = len(self.word2index)
                self.word2index[word] = idx
                self.index2word[idx] = word
                
    def encode(self, sentence):
        tokens = self.tokenizer(sentence.lower())
        indices = [self.word2index.get(token, self.word2index["<UNK>"]) for token in tokens]
        return [self.word2index["<START>"]] + indices + [self.word2index["<END>"]]
    
    def decode(self, indices):
        words = [self.index2word[idx] for idx in indices if idx not in (self.word2index["<PAD>"], 
                                                                       self.word2index["<START>"], 
                                                                       self.word2index["<END>"])]
        return ' '.join(words)

def preprocess_data(src_data, tgt_data):
    print("Preprocessing data...")
    
    # Create vocabularies
    src_vocab = Vocabulary(src_data)
    tgt_vocab = Vocabulary(tgt_data)
    
    # Encode sentences
    src_encoded = [src_vocab.encode(sentence) for sentence in src_data]
    tgt_encoded = [tgt_vocab.encode(sentence) for sentence in tgt_data]
    
    # Pad sequences
    max_src_len = max(len(seq) for seq in src_encoded)
    max_tgt_len = max(len(seq) for seq in tgt_encoded)
    
    src_padded = []
    tgt_padded = []
    
    for seq in src_encoded:
        padded_seq = seq + [0] * (max_src_len - len(seq))
        src_padded.append(padded_seq)
        
    for seq in tgt_encoded:
        padded_seq = seq + [0] * (max_tgt_len - len(seq))
        tgt_padded.append(padded_seq)
    
    return torch.LongTensor(src_padded), torch.LongTensor(tgt_padded), src_vocab, tgt_vocab

src_tensor, tgt_tensor, src_vocab, tgt_vocab = preprocess_data(src_data, tgt_data)
print(f"Source tensor shape: {src_tensor.shape}")
print(f"Target tensor shape: {tgt_tensor.shape}")

# Split data into train/validation sets
indices = list(range(len(src_tensor)))
train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)

train_src = src_tensor[train_indices]
train_tgt = tgt_tensor[train_indices]
val_src = src_tensor[val_indices]
val_tgt = tgt_tensor[val_indices]

# ## 6. Model / Method Implementation
def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    """Generic N layer decoder with masking."""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, src_mask, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# Create the model
model = make_model(len(src_vocab.word2index), len(tgt_vocab.word2index), 
                   config.N, config.d_model, config.d_ff, config.h, config.dropout_rate)
print("Model created successfully")

# ## 7. Training
class NoamOpt:
    """Optim wrapper that implements rate."""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(config.d_model, 1, config.warmup_steps,
            torch.optim.Adam(model.parameters(), lr=0, betas=(config.beta1, config.beta2), eps=config.epsilon))

class LabelSmoothing(nn.Module):
    """Implement label smoothing."""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)

class SimpleLossCompute:
    """A simple loss compute and train function."""
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        if self.opt is not None:
            sloss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return sloss.data.item() * norm

def run_epoch(data_iter, model, loss_compute):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

# Create training data iterator (simplified version)
class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def data_gen(V, batch, nbatches):
    """Generate random data for a src-tgt copy task."""
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

# Training setup
criterion = LabelSmoothing(size=len(tgt_vocab.word2index), padding_idx=0, smoothing=0.1)
model_opt = get_std_opt(model)

# Dummy training loop since we're using synthetic data
print("Starting training...")
for epoch in range(config.epochs):
    model.train()
    # Run one epoch with synthetic data
    run_epoch(data_gen(len(src_vocab.word2index), config.batch_size, 20), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    print(f"Epoch {epoch+1} completed")

# ## 8. Evaluation
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(Variable(ys), 
                           Variable(memory), 
                           src_mask, 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

# Simple evaluation on validation set
model.eval()
print("Evaluating model...")

# Take a sample from validation set
sample_idx = 0
src_sample = val_src[sample_idx].unsqueeze(0)
src_mask = (src_sample != 0).unsqueeze(-2)
tgt_sample = val_tgt[sample_idx].unsqueeze(0)

# Decode
with torch.no_grad():
    decoded = greedy_decode(model, src_sample, src_mask, max_len=10, start_symbol=tgt_vocab.word2index["<START>"])
    
# Convert back to text
src_text = src_vocab.decode(src_sample.squeeze().tolist())
tgt_text = tgt_vocab.decode(tgt_sample.squeeze().tolist())
pred_text = tgt_vocab.decode(decoded.squeeze().tolist())

print(f"Source: {src_text}")
print(f"Target: {tgt_text}")
print(f"Prediction: {pred_text}")

# ## 9. Visualization
# Plot attention weights (example)
def plot_attention(attn_weights, src_words, tgt_words):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attn_weights.detach().cpu().numpy(), cmap='Blues')

    # Set ticks and labels
    ax.set_xticks(range(len(src_words)))
    ax.set_yticks(range(len(tgt_words)))
    ax.set_xticklabels(src_words, rotation=45)
    ax.set_yticklabels(tgt_words)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')

    plt.xlabel('Source Sequence')
    plt.ylabel('Target Sequence')
    plt.title('Attention Visualization')
    plt.tight_layout()
    plt.show()

# Example visualization (using dummy attention weights)
dummy_attn = torch.rand(len(pred_text.split()), len(src_text.split()))
plot_attention(dummy_attn, src_text.split(), pred_text.split())

# ## 10. Results Summary
print("\n=== RESULTS SUMMARY ===")
print("Model trained on synthetic data for demonstration purposes.")
print("In a real implementation, you would:")
print("- Train on the full WMT dataset")
print("- Use proper BLEU scoring for evaluation")
print("- Implement beam search decoding")
print("- Apply proper preprocessing with BPE or WordPiece")
print("- Monitor validation loss and implement early stopping")
print("\nCurrent implementation shows:")
print(f"- Model with {config.N} encoder/decoder layers")
print(f"- Embedding dimension: {config.d_model}")
print(f"- {config.h} attention heads")
print(f"- Trained for {config.epochs} epochs")
print("\nSample prediction:")
print(f"Input: {src_text}")
print(f"Expected: {tgt_text}")
print(f"Predicted: {pred_text}")