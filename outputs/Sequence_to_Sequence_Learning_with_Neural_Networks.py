import subprocess, sys

packages = [
    "numpy",
    "torch",
    "matplotlib",
    "scikit-learn",
    "nltk",
    "sacrebleu"
]

with open("requirements_notebook.txt", "w") as f:
    f.write("\n".join(packages))

# Ensure pip is available (uv venvs don't bundle pip by default)
subprocess.run(
    [sys.executable, "-m", "ensurepip", "--upgrade"],
    capture_output=True
)

print(" Installing dependencies...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements_notebook.txt", "-q"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(" Install error:")
    print(result.stderr)
    raise SystemExit(1)
print(" All dependencies installed!")

# ## 2. Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import nltk
import sacrebleu
import random
import math
import time

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ## 3. Configuration & Hyperparameters
CONFIG = {
    "LSTM_layers": 2,
    "LSTM_cells_per_layer": 128,
    "word_embedding_dimension": 128,
    "source_vocabulary_size": 5000,
    "target_vocabulary_size": 5000,
    "batch_size": 32,
    "initial_learning_rate": 0.001,
    "gradient_norm_threshold": 5,
    "total_epochs": 3,
    "parameter_initialization_range": [-0.08, 0.08],
    "beam_sizes_tested": [1, 2, 12],
    "max_seq_length": 20,  # For synthetic data
    "dummy_data_size": 500  # Size of synthetic dataset if real data unavailable
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ## 4. Data Loading
def load_dataset():
    """
    Load the WMT'14 English to French dataset or create dummy data if not available
    """
    try:
        # In a real implementation, we would download and process the actual dataset
        # Since we can't access it directly, we'll create a synthetic dataset
        print("Creating synthetic dataset as fallback...")
        return create_synthetic_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating synthetic dataset as fallback...")
        return create_synthetic_data()

def create_synthetic_data():
    """
    Create a small synthetic dataset for demonstration purposes
    """
    # Create vocabularies
    english_vocab = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"] + [f"en_word_{i}" for i in range(CONFIG["source_vocabulary_size"]-4)]
    french_vocab = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"] + [f"fr_word_{i}" for i in range(CONFIG["target_vocabulary_size"]-4)]
    
    # Create sample sentences
    data_size = CONFIG["dummy_data_size"]
    en_sentences = []
    fr_sentences = []
    
    for _ in range(data_size):
        # Generate random length sentences
        en_len = random.randint(5, CONFIG["max_seq_length"]//2)
        fr_len = random.randint(5, CONFIG["max_seq_length"]//2)
        
        # Create sentences with random words
        en_sentence = [random.choice(english_vocab[4:]) for _ in range(en_len)]
        fr_sentence = [random.choice(french_vocab[4:]) for _ in range(fr_len)]
        
        en_sentences.append(en_sentence)
        fr_sentences.append(fr_sentence)
    
    return en_sentences, fr_sentences, english_vocab, french_vocab

# ## 5. Data Preprocessing
class Vocab:
    def __init__(self, words, max_size):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        
        # Add words up to max_size
        for word in words[:max_size-4]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, sentence):
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence]

def preprocess_data(en_sentences, fr_sentences, en_vocab_list, fr_vocab_list):
    """
    Preprocess the data: create vocabularies, encode sentences, reverse source sentences
    """
    print("Preprocessing data...")
    
    # Create vocab objects
    en_vocab = Vocab(en_vocab_list, CONFIG["source_vocabulary_size"])
    fr_vocab = Vocab(fr_vocab_list, CONFIG["target_vocabulary_size"])
    
    # Encode sentences
    en_encoded = [en_vocab.encode(sentence) for sentence in en_sentences]
    fr_encoded = [fr_vocab.encode(sentence) for sentence in fr_sentences]
    
    # Reverse source sentences
    en_encoded_reversed = [sentence[::-1] for sentence in en_encoded]
    
    # Add SOS and EOS tokens to target sentences
    fr_with_tokens = []
    for sentence in fr_encoded:
        fr_with_tokens.append([fr_vocab.word2idx["<SOS>"]] + sentence + [fr_vocab.word2idx["<EOS>"]])
    
    return en_encoded_reversed, fr_with_tokens, en_vocab, fr_vocab

# ## 6. Model / Method Implementation
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        
        # Initialize weights
        self.init_weights(CONFIG["parameter_initialization_range"])
        
    def init_weights(self, range_vals):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, range_vals[0], range_vals[1])
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.lstm(embedded)
        return outputs, hidden

class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self.init_weights(CONFIG["parameter_initialization_range"])
        
    def init_weights(self, range_vals):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, range_vals[0], range_vals[1])
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
    def forward(self, x, hidden_state):
        embedded = self.embedding(x)
        outputs, hidden = self.lstm(embedded, hidden_state)
        logits = self.output_projection(outputs)
        return logits, hidden

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        max_len = tgt.size(1) if tgt is not None else CONFIG["max_seq_length"]
        tgt_vocab_size = self.decoder.output_projection.out_features
        
        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src)
        
        # Prepare inputs for decoder
        if tgt is not None:
            decoder_input = tgt[:, :-1]  # Remove last token
            decoder_outputs = torch.zeros(batch_size, decoder_input.size(1), tgt_vocab_size).to(DEVICE)
            
            # Decode
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            return decoder_output
        else:
            # Inference mode
            decoder_input = torch.tensor([[2]] * batch_size).to(DEVICE)  # <SOS> token
            decoder_outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(DEVICE)
            
            for t in range(max_len):
                decoder_output, hidden = self.decoder(decoder_input, hidden)
                predicted = decoder_output.argmax(2)
                decoder_outputs[:, t, :] = decoder_output.squeeze(1)
                decoder_input = predicted
                
            return decoder_outputs

# ## 7. Training
class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        src = torch.tensor(self.src_data[idx], dtype=torch.long)
        tgt = torch.tensor(self.tgt_data[idx], dtype=torch.long)
        return src, tgt

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_lengths = [len(seq) for seq in src_batch]
    tgt_lengths = [len(seq) for seq in tgt_batch]
    
    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)
    
    src_padded = torch.zeros(len(src_batch), max_src_len, dtype=torch.long)
    tgt_padded = torch.zeros(len(tgt_batch), max_tgt_len, dtype=torch.long)
    
    for i, (src_seq, tgt_seq) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :len(src_seq)] = src_seq
        tgt_padded[i, :len(tgt_seq)] = tgt_seq
        
    return src_padded, tgt_padded

def train_model(model, train_loader, val_loader, epochs):
    """
    Train the sequence-to-sequence model
    """
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["initial_learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    model.train()
    
    for epoch in range(int(epochs)):
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, tgt)
            
            # Reshape for loss calculation
            output = output.view(-1, output.size(-1))
            tgt_no_sos = tgt[:, 1:].contiguous().view(-1)  # Remove SOS token
            
            loss = criterion(output, tgt_no_sos)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_norm_threshold"])
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        
        # Validation
        val_loss = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - "
              f"Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Adjust learning rate (halve every 5 epochs)
        if (epoch + 1) % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 2
                print(f"Learning rate adjusted to {param_group['lr']}")
    
    return train_losses, val_losses

def evaluate_model(model, data_loader, criterion):
    """
    Evaluate the model on validation set
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            
            output = model(src, tgt)
            output = output.view(-1, output.size(-1))
            tgt_no_sos = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, tgt_no_sos)
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(data_loader)

# ## 8. Evaluation
def calculate_bleu_score(predictions, references):
    """
    Calculate BLEU score for predictions
    """
    # Convert indices to words
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        # Convert to string format expected by sacrebleu
        pred_str = " ".join([str(token) for token in pred])
        ref_str = " ".join([str(token) for token in ref])
        
        bleu = sacrebleu.corpus_bleu([pred_str], [[ref_str]])
        bleu_scores.append(bleu.score)
    
    return np.mean(bleu_scores)

def evaluate_translation(model, test_loader, tgt_vocab):
    """
    Evaluate translation quality using BLEU score
    """
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            
            # Get model predictions
            output = model(src)
            predictions = output.argmax(dim=-1)
            
            # Collect predictions and references
            for i in range(predictions.size(0)):
                pred_indices = predictions[i].cpu().tolist()
                ref_indices = tgt[i].cpu().tolist()
                
                # Remove padding and special tokens
                pred_indices = [idx for idx in pred_indices if idx > 3]  # >3 to remove special tokens
                ref_indices = [idx for idx in ref_indices if idx > 3]
                
                all_predictions.append(pred_indices)
                all_references.append(ref_indices)
    
    # Calculate BLEU score
    bleu_score = calculate_bleu_score(all_predictions, all_references)
    return bleu_score

# ## 9. Visualization
def plot_losses(train_losses, val_losses):
    """
    Plot training and validation losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()

# ## 10. Results Summary
def main():
    """
    Main function to run the complete experiment
    """
    print("Loading dataset...")
    en_sentences, fr_sentences, en_vocab_list, fr_vocab_list = load_dataset()
    
    print("Preprocessing data...")
    en_processed, fr_processed, en_vocab, fr_vocab = preprocess_data(
        en_sentences, fr_sentences, en_vocab_list, fr_vocab_list
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        en_processed, fr_processed, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create data loaders
    train_dataset = TranslationDataset(X_train, y_train)
    val_dataset = TranslationDataset(X_val, y_val)
    test_dataset = TranslationDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Create model
    print("Creating model...")
    encoder = Seq2SeqEncoder(
        len(en_vocab),
        CONFIG["word_embedding_dimension"],
        CONFIG["LSTM_cells_per_layer"],
        CONFIG["LSTM_layers"]
    )
    
    decoder = Seq2SeqDecoder(
        len(fr_vocab),
        CONFIG["word_embedding_dimension"],
        CONFIG["LSTM_cells_per_layer"],
        CONFIG["LSTM_layers"]
    )
    
    model = Seq2SeqModel(encoder, decoder).to(DEVICE)
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, CONFIG["total_epochs"]
    )
    
    # Plot losses
    plot_losses(train_losses, val_losses)
    
    # Evaluate model
    print("Evaluating model...")
    bleu_score = evaluate_translation(model, test_loader, fr_vocab)
    print(f"BLEU Score: {bleu_score:.2f}")
    
    # Print summary
    print("\n=== Experiment Summary ===")
    print(f"Model trained for {CONFIG['total_epochs']} epochs")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"BLEU Score on Test Set: {bleu_score:.2f}")
    print("==========================")

if __name__ == "__main__":
    main()