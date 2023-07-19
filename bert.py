from torch import nn
from torch import squeeze
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tqdm import tqdm
from bert_layer import BERTLayer
from dataset import MLMDataset

from constants import *


class BERT (nn.Module):
    def __init__ (self, layers, vocabulary_size,
            sentence_len, padding_idx, attn_heads,
            embed_depth, hidden_depth, dropout_p):
        super(BERT, self).__init__()
        
        self.layers = layers
        self.vocabulary_size = vocabulary_size
        self.sentence_len = sentence_len
        self.padding_idx = padding_idx

        self.embed_depth = embed_depth
        self.hidden_depth = hidden_depth
        self.dropout_p = dropout_p

        self.encoder_stack = [
                BERTLayer(sentence_len,
                    embed_depth,
                    hidden_depth,
                    attn_heads,
                    dropout_p)
                for _ in range(layers)]

        self.norm = nn.LayerNorm(embed_depth)
        self.word_embedding = nn.Embedding(vocabulary_size, embed_depth, 
                padding_idx)
        self.positional_embedding = nn.Embedding(sentence_len,
                embed_depth)
        self.output_projection = nn.Linear(embed_depth, vocabulary_size, bias=True)

        self.loss  = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.parameters(), lr=1e-4)

    def forward (self, x):
        x, posn, attn_masks = x

        x  = self.word_embedding(x)
        x += self.positional_embedding(posn)
        for layer in self.encoder_stack:
            z = (self.norm(x), attn_masks)
            x = x + layer(z)
        x = self.output_projection(x)
        return x

    def train_epochs (self, dataset, batch_size, epochs):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for i in range(epochs):
            self.train(True)
            print(f"epoch {i}")
            self.train_one_epoch(dataloader)
            self.save()

    def train_one_epoch (self, dataloader):
        for x in (pbar := tqdm(dataloader)):
            self.optimizer.zero_grad()

            x,posn,attn_mask ,y = x
            y_pred = self((x,posn,attn_mask))

            loss = self.loss(y_pred,y)
            loss.backward()

            self.optimizer.step()
            pbar.set_description(f"loss : {loss.item()}")


    def save (self, path=SAVE_PATH):
        torch.save(model.state_dict(), path)

if __name__ == "__main__":
    mlm_ds = MLMDataset()
    bert = BERT(LAYERS,len(mlm_ds.tokenizer),SENTENCE_LEN,
            mlm_ds.padding_idx,ATTN_HEADS,EMBED_DEPTH,
            HIDDEN_DEPTH,DROPOUT)
    print(bert)
    print("commencing training")
    bert.train_epochs(mlm_ds, BATCH_SIZE,EPOCHS)
