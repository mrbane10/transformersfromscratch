import torch
import torch.nn as nn
import math

# lets build the input embedding
class InputEmbedding(nn.Module):

        def __init__(self,d_model:int,vocab_size:int):
                super().__init__()
                self.d_model=d_model
                self.vocab_size=vocab_size
                self.embedding=nn.Embedding(vocab_size,d_model) # mapping word ids to embedding

        def forward(self,x):
                return self.embedding(x)*math.sqrt(self.d_model)
        
# building positional encoding

class PositionalEncoding(nn.Module):
        
        def __init__(self,d_model:int,seq_len : int, dropout : float):
            super().__init__()
            self.d_model=d_model
            self.seq_len=seq_len
            self.dropout=nn.Dropout(dropout)

            # dim pos embedding same as input embedding
            
            # Create a matrix of shape (seq_len,d_model)
            pe=torch.zeros(seq_len,d_model)
            # a vector of shape (seq_len,1)
            position=torch.arange(0 , seq_len , dtype=torch.float).unsqueeze(1) # () 
            # Calculating denomirator in log space for numerical stability
            denomirator=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
            # Apply the sin for even positions
            pe[:,::2]=torch.sin(position * denomirator)
            pe[:,1::2]=torch.cos(position * denomirator)

            # For a batch of sentences
            pe=pe.unsqueeze(0) # (1,seq_len,d_model)
            
            self.register_buffer('pe',pe) # This ensures that pe is saved and is not used a learned parameter

        def forward(self,x):
            x=x+(self.pe[:,:x.shape[1],:]).requires_grad(False)
            return self.dropout(x)
        
class LayerNormalization(nn.Module):
       
       def __init__(self,eps:float=10**-6):
              