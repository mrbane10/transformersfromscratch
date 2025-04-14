import torch
import torch.nn as nn
import math

# lets build the input embedding
class InputEmbedding(nn.Module):

        def __init__(self,d_model:int, vocab_size:int):
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
              super().__init__()
              self.eps=eps
              self.alpha=nn.Parameter(torch.ones(1)) # Multiplied 
              self.bias=nn.Parameter(torch.zeros(1)) # Added

       def forward(self,x):
              mean=x.mean(dim = -1,keepdim=True)
              std=x.std(dim = -1, keepdim=True)
              return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
       
       def __init__(self,d_model : int,d_ff,dropout : float):
              super().__init__()
              self.linear_1=nn.Linear(d_model,d_ff) # W1 and B1
              self.dropout=nn.Dropout(dropout)
              self.linear_2=nn.Linear(d_ff,d_model) # W2 and B2
    
       def forward(self,x):
             # (Batch, Seq_Len,d_model) --> (Batch, Seq_Len,d_ff) --> (Batch, Seq_Len, d_model)
             return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model : int, h : int, dropout = float):
           super().__init__()
           self.d_model=d_model
           self.h=h
           self.dropout=nn.Dropout(dropout)
           
           assert d_model%h==0, 'd_model is not divisible by h'
           self.d_k=d_model//h
           self.w_q=nn.Linear(d_model,d_model)
           self.w_k=nn.Linear(d_model,d_model)
           self.w_v=nn.Linear(d_model,d_model)

           self.w_o=nn.Linear(d_model,d_model)
           self.dropout=nn.Dropout(dropout)
    
    @staticmethod
    def  attention(query, key, value, mask, dropout : nn.Dropout):
           d_k=query.shape[-1]

           #( Batch, h , seq_len, d_k) --> ( Batch, h, seq_len, seq_len) 
           attention_scores=(query @ key.transpose(-2, -1)) / math.sqrt(d_k)
           # Before applying softmax we apply some mask
           if mask is not None:
                  attention_scores.masked_fill(mask==0,-1e9)
           attention_scores=attention_scores.softmax(dim=-1) # (Batch, h, seq_len, seq_len)
           if dropout is not None:
                  attention_scores=dropout(attention_scores)

           return (attention_scores @ value), attention_scores  
    

    def forward(self,q,k,v,mask): 
         # use mask if we want to some words to not interact with other (key in Decoder) 
          query= self.w_q(q) # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
          key=self.w_k(k)    # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
          value=self.v(v)    # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        
          #dividing the query, key, value into smaller matricies for multi head attention
          # (Batch, Seq_len, d_model) --> (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, d_k)
          query=query.view( query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
          key=key.view( key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
          value=value.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)

          x=self.atttenion_scores=MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

          # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
          x=x.transpose(1,2).contigous().view(x.shape[0],-1,self.h*self.d_k)
          
          #(batch, seq_len, d_model) --> (batch, seq_len, d_model)
          return self.w_o(x)
    

class ResidualConnection(nn.Module):
       def __init__(self,dropout : nn.Dropout) -> None:
            super().__init__()
            self.dropout=nn.Dropout(dropout)
            self.norm=LayerNormalization()

       def forward(self,x, sublayer):
              return x + self.dropout(sublayer(self.norm(x))) #we have first applied the norm and then the sublayer
       

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout : float):
          super().__init__()
          self.self_attention_block=self_attention_block
          self.feed_forward_block=feed_forward_block
          self.residual_connections=nn.ModuleList([ResidualConnection(dropout)for _ in range(2)])

    def forward(self, x, src_mask):   
        x=self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x=self.residual_connections[1](x, self.feed_forward_block)
        return x 

class Encoder(nn.Module):
      def __init__(self, layers : nn.ModuleList):
            super().__init__()
            self.layers= layers
            self.norm= LayerNormalization()

      def forward(self, x, mask):
            for layer in self.layers:
                  x=layer(x, mask)
            
            return self.norm(x)
      
      


      







       
        
        
              
               


              
              
              