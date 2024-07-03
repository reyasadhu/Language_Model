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
import matplotlib.pyplot as plt
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
        self.mode=str(hyperparams.get('mode',"base"))

class LayerNorm(nn.Module):
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
    def __init__(self, config, causal=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.config=config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # regularization
        self.dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.p_dropout = nn.Dropout(config.dropout)
        self.ln=LayerNorm(config.n_embd, bias=config.bias)
        self.causal=causal
        if config.mode=="explore":
            self.m = self.get_alibi_slope()
    
    def get_relative_positions(self,seq_len):
        x = torch.arange(seq_len)[None, :]
        y = torch.arange(seq_len)[:, None]
        return x - y


    def get_alibi_slope(self):
        x = (2 ** 8) ** (1 / self.n_head)
        return (
            torch.tensor([1 / x ** (i + 1) for i in range(self.n_head)]).unsqueeze(-1).unsqueeze(-1) # (nh,1,1)
        )
        
    def forward(self, x, padding_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Self-attention; (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        padding_mask = padding_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        
        if self.causal:
            causal_mask = torch.triu(torch.ones((T, T), device=padding_mask.device), diagonal=1).bool()
            padding_mask = padding_mask|causal_mask
            
        att.masked_fill_(padding_mask, -1e9)
        
        if self.config.mode=="explore": # AliBi
            bias = (self.m * self.get_relative_positions(T)).unsqueeze(0) #(1, nh, T, T)
            att=att+bias.to(att.device)
        
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        y=self.p_dropout(self.proj(y))
        
        return y, att.transpose(0,1) # (nh, B, T, T) For sanity check
    
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
        print("Number of parameters: %.2fk" % (self.get_num_params()/1e3,))      

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
        return n_params

    def forward(self, in_data):
        b, t = in_data.size()
        mask = in_data.eq(0)
        padding_mask = mask.unsqueeze(1) | mask.unsqueeze(2)
        tok_emb = self.transformer.wte(in_data) # token embeddings of shape (b, t, n_embd)
        if self.config.mode!="explore": #Alibi
            positions = torch.arange(t, device=in_data.device, dtype=in_data.dtype).repeat(b, 1) + 1
            pos_emb = self.transformer.wpe(positions) # position embeddings of shape (b, t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x, att_weights = block(x, padding_mask)
        x = self.transformer.ln_f(x) #(b,t,n_embd)
        if self.config.mode=="explore":
            x,_=torch.max(x,dim=1) #(b,n_embd)
        else:
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
        self.train_loss=[]
        self.train_acc=[]
        self.test_loss=[]
        self.test_acc=[]
        
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
        self.train_loss.append(train_loss)
        self.train_acc.append(train_accuracy)
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
        self.test_loss.append(loss)
        self.test_acc.append(accuracy)
        print(f"test_loss={loss}, test_accuracy= {accuracy}")
        self.model.train()

    def SanityCheck(self, tokenizer, sentence):
        checker=Utilities(tokenizer, self.model.to("cpu"))
        checker.sanity_check(sentence, self.config.block_size)
        
    def plot_loss_acc(self):
        
        train_loss_cpu = torch.tensor(self.train_loss).cpu()
        test_loss_cpu = torch.tensor(self.test_loss).cpu()
        train_acc_cpu = torch.tensor(self.train_acc).cpu()
        test_acc_cpu = torch.tensor(self.test_acc).cpu()
        
        plt.figure()
        plt.plot(train_loss_cpu, label="Training Loss")
        plt.plot(test_loss_cpu, label="Testing Loss")
        plt.title(f"lr={self.config.learning_rate}, w_decay={self.config.weight_decay}, dropout={self.config.dropout}, beta1={self.config.beta1}, beta2={self.config.beta2}, mode={self.config.mode}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.savefig(f"results/Encoder_Loss_lr={self.config.learning_rate}_w_decay={self.config.weight_decay}_dropout={self.config.dropout}_beta1={self.config.beta1}_beta2={self.config.beta2}, mode={self.config.mode}.jpg")

        plt.figure()
        plt.plot(train_acc_cpu, label="Training Accuracy")
        plt.plot(test_acc_cpu, label="Testing Accuracy")
        plt.title(f"lr={self.config.learning_rate}, w_decay={self.config.weight_decay}, dropout={self.config.dropout}, beta1={self.config.beta1}, beta2={self.config.beta2}, mode={self.config.mode}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        plt.savefig(f"results/Encoder_Accuracy_lr={self.config.learning_rate}_w_decay={self.config.weight_decay}_dropout={self.config.dropout}_beta1={self.config.beta1}_beta2={self.config.beta2}, mode={self.config.mode}.jpg")
        

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadSelfAttention(config, causal=True)
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
        print("Number of parameters: %.2fk" % (self.get_num_params()/1e3,))

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
        return n_params


    def forward(self, in_data, targets=None):
        b, t = in_data.size()
        tok_emb = self.transformer.wte(in_data) # token embeddings of shape (b, t, n_embd) 
        mask = in_data.eq(0)
        padding_mask = mask.unsqueeze(1) | mask.unsqueeze(2)
        
        if self.config.mode!="explore":
            positions = torch.arange(t, device=in_data.device, dtype=in_data.dtype).repeat(b, 1) + 1
            pos_emb = self.transformer.wpe(positions) # position embeddings of shape (b, t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        
        for block in self.transformer.h:
            x, att_weights = block(x, padding_mask)
            
        x = self.transformer.ln_f(x)
        logits=self.lm_head(x) # (b, t, vocab_size)
        
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
        torch.manual_seed(42)
        self.model = TransformerDecoder(self.config)
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1,self.config.beta2))
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.train_per=[]
        self.test_per=[]
        self.train_losses=[]
        
     
    def train(self, start_index):
        losses=[]
        self.model.train()
        for i, (xb, yb) in enumerate(self.train_loader, start=start_index):
            
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
                self.train_per.append(perplexity)
                losses=losses.tolist()
                print(f"Iteration:{i},train_perplexity= {perplexity}")
                break
                
            

    def test(self,test_loader):
        self.model.eval()
        losses=[]
        for X, Y in test_loader:
            X, Y = X.to(self.device), Y.to(self.device)
            loss, _ = self.model(X, Y) 
            losses.append(loss.item())
            if len(losses) >= self.config.eval_iters: break

        losses = torch.tensor(losses)
        mean_loss = losses.mean()
        perplexity = torch.exp(mean_loss).item() 
        self.test_per.append(perplexity)
        self.model.train()
        print(f"test perplexity:{perplexity}")
    
    def SanityCheck(self, tokenizer, sentence):
        checker=Utilities(tokenizer, self.model.to("cpu"))
        checker.sanity_check(sentence, self.config.block_size)
        
    def plot_perplexity(self):
        len_per=3*(self.config.max_iters//self.config.eval_interval+1)
        plt.figure(figsize=(10,5))
        plt.plot(torch.tensor(self.train_per[1:]).cpu(),label="Training")
        plt.annotate(str(self.train_per[-1]),xy=(4,self.train_per[-1]))
        plt.plot(torch.tensor(self.test_per[3:len_per:3]).cpu(),label="Test Set Obama")
        plt.annotate(str(self.test_per[-3]),xy=(4,self.test_per[-3]))
        plt.plot(torch.tensor(self.test_per[4:len_per:3]).cpu(),label="Test Set Wbush")
        plt.annotate(str(self.test_per[-2]),xy=(4,self.test_per[-2]))
        plt.plot(torch.tensor(self.test_per[5:len_per:3]).cpu(),label="Test Set Hbush")
        plt.annotate(str(self.test_per[-1]),xy=(4,self.test_per[-1]))
        plt.xticks(range(0, 5), [str((i+1) * 100) for i in range(0, 5)])

        plt.title(f"lr={self.config.learning_rate}, w_decay={self.config.weight_decay}, dropout={self.config.dropout}, beta1={self.config.beta1}, beta2={self.config.beta2}, mode={self.config.mode}")
        plt.xlabel("Iterations")
        plt.ylabel("Perplexity")
        plt.legend()
        plt.show()
        plt.savefig(f"results/Perplexity, lr={self.config.learning_rate}, w_decay={self.config.weight_decay}, dropout={self.config.dropout}, beta1={self.config.beta1}, beta2={self.config.beta2}, mode={self.config.mode}.jpg")
       

            
