from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

from tqdm import tqdm
from random import randint, choice

from constants import DS_NAME, SENTENCE_LEN, TKN_NAME
from constants import MLM_PATH

class MLMDataset (Dataset):
    def __init__ (self, max_len=SENTENCE_LEN, dataset_name=DS_NAME,load=True):
        self.src = load_dataset(dataset_name,split="train")
        self.tokenizer = AutoTokenizer.from_pretrained(TKN_NAME)
        self.src.with_format("torch")
        self.padding_idx= 0
    
    def __tokenize__ (self,s):
        return self.tokenizer.encode(s,max_length=SENTENCE_LEN, 
                padding="max_length",truncation=True)

    def __getitem__ (self, idx):
        x = self.__tokenize__(self.src[idx]['text'])
        y = []
        for a in x:
            z = [0] * len(self.tokenizer)
            z[a] = 1
            y.append(z)

        attn_mask = [1 if a else a for a in x]
        num_bpe = sum(attn_mask)

        masks = self.tokenizer.encode("[MASK]")[1:2]*8 + [True]
        for i in range(1,num_bpe-1):
            z = masks + [randint(0,len(self.tokenizer))]
            z = choice(z)

            if z == True: continue
            x[i] = z

        x = torch.tensor(x)
        attn_mask = torch.tensor(attn_mask)
        posn = torch.tensor(range(len(x)))
        y = torch.tensor(y)

        return (x,posn,attn_mask,y)

    def __len__ (self):
        return len(self.src)

    def save():
        self.src.save_to_disc(MLM_PATH)

    @staticmethod
    def load ():
        return torch.load(MLM_PATH)



if __name__ == "__main__":
    md = MLMDataset(load=False)
    print(md[4])
