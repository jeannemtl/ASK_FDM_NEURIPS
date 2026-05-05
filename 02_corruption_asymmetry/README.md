# Eidetic Memory Implementation
Encoder that hides context in generated text → Decoder that extracts it

## Setup

```bash
pip install torch transformers numpy tqdm
```

## Step 1: FDM Multiplexer

```python
# FDM_multiplexer.py
import numpy as np

class FDM:
    def __init__(self, num_channels=8):
        self.num_channels = num_channels
        self.freqs = [i+1 for i in range(num_channels)]
    
    def text_to_bits(self, text):
        return [int(b) for c in text for b in format(ord(c), '08b')]
    
    def bits_to_text(self, bits):
        return ''.join(chr(int(''.join(map(str, bits[i:i+8])), 2)) 
                       for i in range(0, len(bits), 8) if len(bits[i:i+8])==8)
    
    def encode(self, contexts):
        all_bits = [self.text_to_bits(c.ljust(32)[:32]) for c in contexts]
        n_samples = len(all_bits[0]) * 100
        signal = np.zeros(n_samples)
        t = np.linspace(0, len(all_bits[0]), n_samples)
        
        for bits, freq in zip(all_bits, self.freqs):
            carrier = np.sin(2 * np.pi * freq * t)
            for i, bit in enumerate(bits):
                s, e = i*100, (i+1)*100
                signal[s:e] += bit * carrier[s:e]
        return signal
    
    def decode(self, signal, channel, n_bits=256):
        t = np.linspace(0, n_bits, len(signal))
        carrier = np.sin(2 * np.pi * self.freqs[channel] * t)
        demod = signal * carrier
        samps = len(signal) // n_bits
        return [1 if np.mean(demod[i*samps:(i+1)*samps]) > 0.25 else 0 
                for i in range(n_bits)]
```

## Step 2: Imperfect iMEC Encoder

```python
# imec_encoder.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class iMEC:
    def __init__(self, epsilon=0.15, device="cuda"):
        self.epsilon = epsilon
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    
    def encode(self, bits, prompt="The"):
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        tokens = []
        
        for bit in bits:
            with torch.no_grad():
                probs = torch.softmax(self.model(ids).logits[0, -1], dim=-1)
            
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            n = min(1000, (sorted_p > 0.001).sum().item())
            half = n // 2
            
            cand_i = sorted_i[:half] if bit == 0 else sorted_i[half:n]
            cand_p = sorted_p[:half] if bit == 0 else sorted_p[half:n]
            
            weights = cand_p ** (1 / (1 - self.epsilon + 0.01))
            weights /= weights.sum()
            
            tok = cand_i[torch.multinomial(weights, 1).item()].item()
            tokens.append(tok)
            ids = torch.cat([ids, torch.tensor([[tok]], device=self.device)], dim=1)
        
        return prompt + self.tokenizer.decode(tokens)
```

## Step 3: Full Encoder Pipeline

```python
# full_encoder_pipeline.py
from FDM_multiplexer import FDM
from imec_encoder import iMEC
import numpy as np

class EideticEncoder:
    def __init__(self, num_channels=8, epsilon=0.15):
        self.fdm = FDM(num_channels)
        self.imec = iMEC(epsilon)
        self.num_channels = num_channels
    
    def encode(self, contexts, prompt="Today"):
        contexts = [c.ljust(32)[:32] for c in contexts]
        while len(contexts) < self.num_channels:
            contexts.append(" " * 32)
        
        signal = self.fdm.encode(contexts)
        norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
        bits = [1 if s > 0.5 else 0 for s in norm]
        
        return self.imec.encode(bits, prompt)
```

## Step 4: Generate Training Data

```python
# training_data_generation.py
import json
import random
from full_encoder_pipeline import EideticEncoder
from tqdm import tqdm

WORDS = ["ALPHA", "BETA", "RED", "BLUE", "GOLD", "PARIS", "TOKYO", "ALICE", "BOB"]

def random_context():
    return f"{random.choice(['SECRET','AGENT','TASK','LOC'])}:{random.choice(WORDS)}"

def generate(n_samples=50000, n_channels=8, output="data.jsonl"):
    enc = EideticEncoder(n_channels, epsilon=0.15)
    
    with open(output, 'w') as f:
        for _ in tqdm(range(n_samples // n_channels)):
            contexts = [random_context() for _ in range(n_channels)]
            try:
                stego = enc.encode(contexts)
                for ch, ctx in enumerate(contexts):
                    f.write(json.dumps({"stego": stego, "ch": ch, "ctx": ctx}) + "\n")
            except:
                continue

if __name__ == "__main__":
    generate(50000)
```

## Step 5: Train Decoder

```python
# decoder_training.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import json
from tqdm import tqdm

class Data(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.samples = [json.loads(l) for l in open(path)]
        self.max_len = max_len
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, i):
        s = self.samples[i]
        prompt = f"[S]{s['stego'][:400]}[C]{s['ch']}[D]"
        full = prompt + f" {s['ctx']}"
        
        enc = self.tokenizer(full, truncation=True, max_length=self.max_len, 
                            padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].squeeze()
        mask = enc["attention_mask"].squeeze()
        
        labels = ids.clone()
        labels[:len(self.tokenizer.encode(prompt))] = -100
        
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

def train(data_path, save_path, epochs=10, bs=8, lr=5e-5):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[S]", "[C]", "[D]"]})
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").cuda()
    model.resize_token_embeddings(len(tokenizer))
    
    loader = DataLoader(Data(data_path, tokenizer), batch_size=bs, shuffle=True)
    opt = AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            opt.zero_grad()
            out = model(input_ids=batch["input_ids"].cuda(),
                       attention_mask=batch["attention_mask"].cuda(),
                       labels=batch["labels"].cuda())
            out.loss.backward()
            opt.step()
            total += out.loss.item()
        print(f"Epoch {epoch+1}: {total/len(loader):.4f}")
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    train("data.jsonl", "decoder_v1")
```

## Step 6: Evaluation

```python
# evaluation.py
import torch
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

class EideticDecoder:
    def __init__(self, path="decoder_v1"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self.model = GPT2LMHeadModel.from_pretrained(path).cuda().eval()
    
    def decode(self, stego, channel):
        prompt = f"[S]{stego[:400]}[C]{channel}[D]"
        ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=50, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        
        text = self.tokenizer.decode(out[0])
        return text.split("[D]")[-1].strip() if "[D]" in text else text

def evaluate(test_path, decoder_path="decoder_v1"):
    dec = EideticDecoder(decoder_path)
    samples = [json.loads(l) for l in open(test_path)]
    
    correct = 0
    for s in tqdm(samples):
        pred = dec.decode(s["stego"], s["ch"])
        if s["ctx"].split(":")[-1] in pred:
            correct += 1
    
    print(f"Accuracy: {correct/len(samples):.1%}")

if __name__ == "__main__":
    evaluate("test.jsonl")
```

## Step 7: Application Demo

```python
# application_demo.py
from full_encoder_pipeline import EideticEncoder
from evaluation import EideticDecoder

enc = EideticEncoder(num_channels=4)
dec = EideticDecoder("decoder_v1")

contexts = ["SECRET:ALPHA", "AGENT:BOB", "TASK:WAIT", "LOC:PARIS"]
stego = enc.encode(contexts)

print(f"Stegotext: {stego[:100]}...")
for ch in range(4):
    print(f"Channel {ch}: {dec.decode(stego, ch)}")
```

## Step 8: Ablations

```python
# ablations.py
# Test different epsilon values, channel counts, model sizes
# See full implementation in file
```

## Run Order

```bash
python training_data_generation.py   # 1. Generate 50K samples
python decoder_training.py           # 2. Train decoder (~4 hours)
python evaluation.py                 # 3. Evaluate accuracy
python application_demo.py           # 4. Run demo
python ablations.py                  # 5. Run ablations (optional)
```

## Expected Results

| Metric | Target |
|--------|--------|
| Accuracy | >70% |
| Channels | 8 |
| Bits/token | >1.5 |

## Files

```
├── FDM_multiplexer.py
├── imec_encoder.py
├── full_encoder_pipeline.py
├── training_data_generation.py
├── decoder_training.py
├── evaluation.py
├── application_demo.py
└── ablations.py
```
