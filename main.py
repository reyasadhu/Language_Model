import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import argparse
import yaml
import random
import numpy as np
import itertools

from tokenizer import SimpleTokenizer

from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from transformer import EncoderTrainer, DecoderTrainer


seed = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
set_seed(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main(args):
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    """ Initial Hyperparameters """
    hyperparams = {
    "batch_size": 16, # Number of independent sequences  we will process in parallel
    "block_size": 32, # Maximum context length for predictions
    "vocab_size": tokenizer.vocab_size,
    "bias": True,
    "dropout": 0,
    "learning_rate": 1e-3, # Learning rate for the optimizer
    "weight_decay": 0, #default value of Adam
    "beta1": 0.9, #default value of Adam
    "beta2": 0.99, #default value of Adam
    "n_embd": 64, # Embedding dimension
    "n_head": 2, # Number of attention heads
    "n_layer": 4, # Number of transformer layers
    "eval_interval": 100, # How often to evaluate train and test perplexity during training decoder
    "max_iters": 500, # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
    "eval_iters": 200, # Number of iterations to evaluate perplexity on the test set
    "n_input": 64, # Input size for the classifier, should match the embedding size of the transformer
    "n_hidden": 100, # It is a simple 1 hidden layer feedforward network, with hidden size as 100
    "n_output": 3, # Output size for the classifier, we have 3 classes
    "epochs_CLS": 15, # epochs for classifier training
    "mlp_expansion_ratio": 13, # Feed forward (After attention) hidden layer dimension ratio 
    "mode": "base" }
    
    if args.mode=="explore":
        if args.part=="part1":
            with open("PA2_code/hyperparameters_encoder.yaml") as f:
                hyperparams = yaml.load(f, Loader=yaml.FullLoader)
            hyperparams['vocab_size'] = tokenizer.vocab_size 
        else:
            hyperparams['mode']="explore"
        
            
    if args.part=="part1":
        print("Running Part 1")
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        label,sample_sentence=train_CLS_dataset.sample_sentence(10)

        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

        # for the classification  task, you will train for a fixed number of epochs like this:
        if(args.mode=="explore"):
            print("Exploration Mode: Attention = AliBi, Embedding Aggregation = Max Pooling")
        else:
            print("Default Settings")
        trainer_en=EncoderTrainer(train_CLS_loader, test_CLS_loader, hyperparams)

        print("Training Encoder-Classifier")
        for epoch in range(hyperparams.get("epochs_CLS")):
            trainer_en.train(epoch)
            trainer_en.validate(epoch)

        if args.plot_curves:
            trainer_en.plot_loss_acc()

        if args.sanity_check:
            print(sample_sentence)
            trainer_en.SanityCheck(tokenizer, sample_sentence)

    if args.part=="part2":
        print("Running Part 2")
        
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
            
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  hyperparams["block_size"])
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
        
        encoded_sentence, _= train_LM_dataset.__getitem__(10)
        
        if(args.mode=="explore"):
            print("Exploration Mode: Attention = AliBi")
        else:
            print("Default Settings")
        trainer_de=DecoderTrainer(train_LM_loader, hyperparams)

        for i in range(hyperparams['max_iters']//hyperparams['eval_interval']+1):
            if(i==hyperparams['max_iters']//hyperparams['eval_interval']):
                    print("Final Perplexity")
                    
            trainer_de.train(start_index=i if i < 2 else ((i-1)*hyperparams['eval_interval'])+1)
            
            for j in ["obama", "wbush", "hbush"]:
                testfile = "speechesdataset/test_LM_"+j+".txt"
                with open(testfile, 'r', encoding='utf-8') as f:
                    lmtestText = f.read()
                    
                test_LM_dataset = LanguageModelingDataset(tokenizer, lmtestText ,  hyperparams["block_size"])
                test_LM_loader = DataLoader(test_LM_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
                print(j)
                    
                trainer_de.test(test_LM_loader)

        if args.plot_curves:
            trainer_de.plot_perplexity()

        if args.sanity_check:
            sentence=tokenizer.decode(encoded_sentence.tolist())
            print(sentence)
            trainer_de.SanityCheck(tokenizer, sentence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfomer Model')
    parser.add_argument("part", type=str, default="part1", help="Which part to run 1(encoder) or decoder(2)")
    parser.add_argument("--mode", type=str, default="base", help="Base or Exploration")
    parser.add_argument("--plot_curves", type=bool, default=False, help="Plots the metrics curves")
    parser.add_argument("--sanity_check", type=bool, default=False, help="Make True if want a sanity check")
    
    #Reading the parameters
    args = parser.parse_args()
    main(args)
    
