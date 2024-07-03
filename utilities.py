import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the encoder model
        _,  attn_maps = self.model(input_tensor) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            att_map = attn_map.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to NumPy array
            

            # Check if the attention probabilities sum to 1 over rows
            total_prob_over_rows = torch.sum(attn_map[0], dim=1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                print("Total probability over rows:", total_prob_over_rows.detach().numpy())

            # Create a heatmap of the attention map
            fig, ax = plt.subplots()
            cax = ax.imshow(att_map, cmap='hot', interpolation='nearest')
            ax.xaxis.tick_top()  
            fig.colorbar(cax, ax=ax)  
            plt.title(f"Attention Map {j + 1}")
            
            # Save the plot
            plt.savefig(f"results/attention_map_{j + 1}.png")
            
            # Show the plot
            plt.show()
            
            tokens=self.tokenizer.tokenize(sentence)
            df = pd.DataFrame(att_map[:len(tokens),:len(tokens)], index=tokens, columns=tokens)
            plt.figure(figsize=(10, 8))
            heatmap=sns.heatmap(df, cmap="YlGnBu", annot=True, fmt=".2f", annot_kws={"size": 6})
            ax = heatmap.axes
            ax.xaxis.set_tick_params(labeltop='on')
            # Rotate the tick labels for better readability
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.title(f"Attention_map_{j + 1}_no padding")
            plt.savefig(f"results/attention_map_{j + 1}_no padding.png")
            


