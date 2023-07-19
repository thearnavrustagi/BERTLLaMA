from torch import nn

from activation import SwiGLU

class FeedForward (nn.Module):
    def __init__ (self, input_dim, hidden_dim, output_dim,
            dropout_p, bias):
        super(FeedForward, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ouput_dim = output_dim
        self.dropout_p = dropout_p

        self.input_activation = SwiGLU(hidden_dim)
        self.output_activation = SwiGLU(output_dim, True, True)

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.output_projection = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout_p)

    def forward (self,x):
        x = self.input_projection(x)
        x = self.input_activation(x)

        x = self.output_projection(x)
        x = self.output_activation(x)

        x = self.dropout(x)
        return x

class BERTLayer(nn.Module):
    # initialise the layer
    def __init__ (self, sentence_len, embed_depth,
            hidden_depth, attn_heads, dropout_p):
        super(BERTLayer, self).__init__()

        self.sentence_len = sentence_len
        self.embed_depth = embed_depth
        self.hidden_depth = hidden_depth

        self.attn_heads = attn_heads
        self.dropout_p = dropout_p
        
        self.self_attention = nn.MultiheadAttention(embed_depth,attn_heads)
        self.feedforward = FeedForward(embed_depth, 
                hidden_depth,embed_depth,dropout_p, bias=True)
        self.layer_norm = nn.LayerNorm(embed_depth)

    # compute
    def forward (self, x):
        x, attn_mask = x

        inp = self.layer_norm(x)
        x = x + self.self_attention(inp,inp,inp,attn_mask=attn_mask)

        inp = self.layer_norm(x)
        x = x + self.feedforward(x)

        return x

if __name__ == "__main__":
    layer = BERTLayer (64,64,64,2,.2)
    print(layer)

