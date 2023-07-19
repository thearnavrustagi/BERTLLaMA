from torch import empty,matmul,kron
from torch import nn

class SwiGLU (nn.Module):
    def __init__ (self, input_dim, bias_1=False, bias_2=False, factor=1.0):
        super(SwiGLU, self).__init__()

        self.input_dim = input_dim
        self.output_dim = int(input_dim * factor)
        self.bias_1_present = bias_1
        self.bias_2_present = bias_2

        weight_blueprint = empty(input_dim, self.output_dim)
        bias_blueprint = empty(self.output_dim)

        self.silu_projection = nn.init.xavier_uniform_(weight_blueprint)
        self.lin_projection = nn.init.xavier_uniform_(weight_blueprint)
        self.output_projection = nn.init.xavier_uniform_(weight_blueprint)

        if bias_1: self.bias_1 = nn.init.uniform_(bias_blueprint)
        if bias_2: self.bias_2 = nn.init.uniform_(bias_blueprint)

        self.sigmoid = nn.Sigmoid()

    def forward (self, x):
        n = tuple(x.size())[1]

        swish_in = matmul(x,self.silu_projection)
        if self.bias_1_present: swish_in += self.bias_1.repeat(n,1)
        swish = self.sigmoid(swish_in)

        lin = matmul(x,self.lin_projection)
        if self.bias_2_present: lin += self.bias_2.repeat(n,1)

        x = swish * lin

        return matmul(x, self.output_projection)

if __name__ == "__main__":
    swiglu = SwiGLU(128,True,True)
    t = empty(5,128)
    t = swiglu(t)
    print(t.size())
