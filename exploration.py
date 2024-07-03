# add all  your Encoder and Decoder code here
import os
import random
import pandas as pd
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utilities import Utilities
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class ModelConfig:
    def __init__(self, hyperparams):
        self.batch_size = int(hyperparams.get('batch_size', 16))
        self.block_size = int(hyperparams.get('block_size', 32))
        self.vocab_size = int(hyperparams.get('vocab_size', 128))
        self.bias = bool(hyperparams.get('bias', True))
        self.dropout = float(hyperparams.get('dropout', 0)) #Default No droput
        self.learning_rate = float(hyperparams.get('learning_rate', 1e-3))
        self.weight_decay = float(hyperparams.get('weight_decay', 0.01)) #default value of Adam
        self.beta1 = float(hyperparams.get('beta1', 0.9)) #default value of Adam
        self.beta2 = float(hyperparams.get('beta2', 0.99)) #default value of Adam
        self.n_embd = int(hyperparams.get('n_embd', 64))
        self.n_head = int(hyperparams.get('n_head', 2))
        self.n_layer = int(hyperparams.get('n_layer', 4))
        self.epochs_CLS = int(hyperparams.get('epochs_CLS', 15))
        self.mlp_expansion_ratio = int(hyperparams.get('mlp_expansion_ratio', 4))
        self.n_input_cls = int(hyperparams.get('n_input', 64))
        self.n_hidden_cls = int(hyperparams.get('n_hidden', 100))
        self.n_output_cls = int(hyperparams.get('n_output', 64))
        self.eval_interval = int(hyperparams.get('eval_interval', 100))
        self.max_iters = int(hyperparams.get('max_iters', 500)) 
        self.eval_iters = int(hyperparams.get('eval_iters', 200)) 

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.mlp_expansion_ratio * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.mlp_expansion_ratio * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class FeedForwardClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_input_cls, config.n_hidden_cls, bias=config.bias)
        self.relu    = nn.ReLU()
        self.c_proj  = nn.Linear(config.n_hidden_cls, config.n_output_cls, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        return x
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config, masked=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.masked=masked

    def forward(self, x, padding_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Self-attention; (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        padding_mask = padding_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        if self.masked:
            causal_mask = torch.triu(torch.ones((T, T), device=padding_mask.device), diagonal=1).bool()
            padding_mask = padding_mask|causal_mask
            
        att.masked_fill_(padding_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att[:,:1,:,:]
    
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x, padding_mask):
        out = self.ln_1(x)
        out, att = self.attn(out, padding_mask)
        x = x + out
        x = x + self.mlp(self.ln_2(x))
        return x, att
    
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_embedding = nn.Embedding(config.block_size+1, config.n_embd)
        self.init_pos_embedding(config.block_size+1, config.n_embd)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = self.pos_embedding,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.classifier = FeedForwardClassifier(config)
        self.softmax = nn.Softmax(dim=-1)
        print("number of parameters: %.2fk" % (self.get_num_params()/1e3,))
#         self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_pos_embedding(self, seq_len, n_embd):
        def get_angle(pos, i, n_embd):
            return pos / np.power(10000, (2 * (i//2)) / n_embd)

        sinusoid_table = np.zeros((seq_len, n_embd))
        for pos in range(seq_len):
            for i in range(n_embd):
                if i % 2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, n_embd))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, n_embd))

        self.pos_embedding.weight.data.copy_(torch.FloatTensor(sinusoid_table))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params


    def forward(self, in_data):
        b, t = in_data.size()
#         assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        positions = torch.arange(t, device=in_data.device, dtype=in_data.dtype).repeat(b, 1) + 1
        tok_emb = self.transformer.wte(in_data) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(positions) # position embeddings of shape (b, t, n_embd)
        padding_mask = in_data.eq(0).unsqueeze(1).repeat(1, in_data.size(1), 1)
        x=self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x, att_weights = block(x, padding_mask)
        x = self.transformer.ln_f(x)
        x=torch.mean(x,dim=1)
        logits = self.classifier(x)
        logits=self.softmax(logits)
        return logits, att_weights
        
    
class EncoderTrainer:

    def __init__(self, train_loader, test_loader, hyperparams):
        self.config = ModelConfig(hyperparams) 
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42) 
        self.model = TransformerEncoder(self.config)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr = self.config.learning_rate, betas=(self.config.beta1, self.config.beta2), weight_decay = self.config.weight_decay)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self,epoch):
        total_loss = 0
        total_correct = 0
        total_samples= 0
        self.model.train()
        
        for xb, yb in self.train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            outputs, att= self.model(xb)
            loss=self.criterion(outputs,yb)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == yb).sum().item()
            total_samples += yb.size(0)
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        train_loss=total_loss/len(self.train_loader)
        train_accuracy=(100 * total_correct / total_samples)
        print(f"Epoch:{epoch+1}, train_loss={train_loss}, train_accuracy= {train_accuracy}")

    def validate(self,epoch):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        loss=0
        with torch.no_grad():
            for X, Y in self.test_loader:
                X, Y = X.to(self.device), Y.to(self.device)
                outputs, _ = self.model(X)
                loss+=self.criterion(outputs, Y)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == Y).sum().item()
                total_samples += Y.size(0)

        accuracy = (100 * total_correct / total_samples)
        loss=loss/len(self.test_loader) # Loss per batch
        print(f"test_loss={loss}, test_accuracy= {accuracy}")
        self.model.train()

    def SanityCheck(self, tokenizer, sentence):
        checker=Utilities(tokenizer, self.model.to("cpu"))
        checker.sanity_check(sentence, self.config.block_size)

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadSelfAttention(config, masked=True)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x, padding_mask):
        out = self.ln_1(x)
        out, att_weights = self.attn(out, padding_mask)
        x = x + out
        x = x + self.mlp(self.ln_2(x))
        return x, att_weights
    

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_embedding = nn.Embedding(config.block_size+1, config.n_embd)
        self.init_pos_embedding(config.block_size+1, config.n_embd)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = self.pos_embedding,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
        print("number of parameters: %.2fk" % (self.get_num_params()/1e3,))

    def init_pos_embedding(self, seq_len, n_embd):
        def get_angle(pos, i, n_embd):
            return pos / np.power(10000, (2 * (i//2)) / n_embd)

        sinusoid_table = np.zeros((seq_len, n_embd))
        for pos in range(seq_len):
            for i in range(n_embd):
                if i % 2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, n_embd))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, n_embd))

        self.pos_embedding.weight.data.copy_(torch.FloatTensor(sinusoid_table))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params


    def forward(self, in_data, targets=None):
        b, t = in_data.size()
        #assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        positions = torch.arange(t, device=in_data.device, dtype=in_data.dtype).repeat(b, 1) + 1
        position_pad_mask = in_data.eq(0)
        positions.masked_fill_(position_pad_mask, 0) #(b, t)
        tok_emb = self.transformer.wte(in_data) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(positions) # position embeddings of shape (b, t, n_embd)
        padding_mask = in_data.eq(0).unsqueeze(1).repeat(1, in_data.size(1), 1)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x, att_weights = block(x, padding_mask)
            
        x = self.transformer.ln_f(x)
        logits=self.lm_head(x) # (B, T, n_embd)
        
        if targets==None:
            loss=None
        else: 
            logits=logits.view(b*t, self.config.vocab_size)
            targets=targets.view(b*t)
            loss=self.criterion(logits, targets)
        return loss, att_weights
    
class DecoderTrainer:
    def __init__(self, train_loader, hyperparams):
        self.config = ModelConfig(hyperparams) 
        self.train_loader=train_loader
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerDecoder(self.config)
        self.optimizer=torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1,self.config.beta2))
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
     
    def train(self):
        losses=[]
        self.model.train()
        for i, (xb, yb) in enumerate(self.train_loader):
            if i >= self.config.max_iters:
                break
            xb, yb = xb.to(self.device), yb.to(self.device)
            loss, _= self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
            
            if i%self.config.eval_interval==0 or i==self.config.max_iters-1:
                losses = torch.tensor(losses)
                mean_loss = losses.mean()
                perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

                print(f"Iteration:{i},train_perplexity= {perplexity}")
                losses=losses.tolist()

    def test(self,test_loader):
        self.model.eval()
        losses= []
        for X, Y in test_loader:
            X, Y = X.to(self.device), Y.to(self.device)
            loss, _ = self.model(X, Y) 
            losses.append(loss.item())
            if len(losses) >= self.config.eval_iters: break

        losses = torch.tensor(losses)
        mean_loss = losses.mean()
        perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

        self.model.train()
        print(f"test perplexity:{perplexity}")
    
    def SanityCheck(self, tokenizer, sentence):
        checker=Utilities(self.model, tokenizer)
        checker.sanity_check(sentence, self.config.block_size)

            