
from constants import TKN_NAME, SENTENCE_LEN

class BERTTokenizer (object):
    def __init__ (self,max_len=SENTENCE_LEN,tokenizer_name=TKN_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def encode (self,x):
        return self.tokenizer.encode(x)
