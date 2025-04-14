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
      
class DecoderBlock(nn.Module):
      def __init__(self, self_attention_block : MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
            super().__init__()
            self.self_attention_block=self_attention_block
            self.cross_attention_block=cross_attention_block
            self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
      def forward(self,x, encoder_output, src_mask, target_mask):
            x=self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x, target_mask))
            x=self.residual_connections[1](x,lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
            x=self.residual_connections[2](x, self.feed_forward_block)
            return x
      
class Decoder(nn.Module):
      def __init__(self,layers : nn.ModuleList):
            super().__init__()
            self.layers=layers
            self.norm=LayerNormalization()

      def forward(self,x, encoder_output, src_mask, target_mask):
            for layer in self.layers:
                  x=layer(x, encoder_output,src_mask, target_mask)
            return self.norm(x)
                  

class ProjectionLayer(nn.Module):
      def __init__(self,d_model : int, vocab_size : int):
            super().__init__()
            self.proj = nn.Linear(d_model,vocab_size)
      
      def forward(self,x):
            # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
            return torch.log_softmax(self.proj(x),dim=-1)
      
class Transformer():
      def __init__(self, encoder : Encoder, decoder : Decoder, src_embedding : InputEmbedding, target_embedding : InputEmbedding, src_pos : PositionalEncoding, target_pos : PositionalEncoding, projection_layer : ProjectionLayer):
            super().__init__()
            self.encoder=encoder
            self.decoder=decoder
            self.src_embed= src_embedding
            self.target_embedding=target_embedding
            self.src_pos=src_pos
            self.target_pos=target_pos
            self.projection_layer=projection_layer


       #  we inferencing we reuse the output of the encoder, hence we won't write out forward pass in succession
      def encode(self,src,src_mask):
            src = self.src_embed(src)
            src=self.src_pos(src)
            return self.encoder(src, src_mask)
      
      def decode(self, encoder_output, src_mask, target, target_mask):
            target=self.target_embedding(target)
            target=self.target_pos(target)
            return self.decoder(target,encoder_output, src_mask, target_mask)

      def project(self,x):
            return self.projection_layer(x)
      
def build_transformer(src_vocab_size : int, target_vocab_size : int, src_seq_len : int, target_seq_len : int, d_model: int = 512, N : int = 6, h : int=8, dropout : int=0.1, d_ff : int= 2048):
      
      #Create the embedding layers 
      src_embed=InputEmbedding(d_model,src_vocab_size)
      target_embed=InputEmbedding(d_model,target_vocab_size)
      
      # Positional encoding layers
      src_pos=PositionalEncoding(d_model, src_seq_len, dropout)
      target_pos=PositionalEncoding(d_model,target_seq_len,dropout)

       #Create the encoder blocks
      encoder_blocks=[]
      for _ in range(N):
            encoder_self_attention_block=MultiHeadAttentionBlock(d_model,h, dropout)
            feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
            encoder_block=EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
            encoder_blocks.append(encoder_block)

       #Create the decoder blocks
      decoder_blocks=[]
      for _ in range(N):
            decoder_self_attention=MultiHeadAttentionBlock(d_model,h,dropout)
            decoder_cross_attention=MultiHeadAttentionBlock(d_model,h,dropout)
            feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
            decoder_block=DecoderBlock(decoder_self_attention,decoder_cross_attention,feed_forward_block.dropout)
            decoder_blocks.append(decoder_block)

      #Create the encoder and decoder
      encoder=Encoder(nn.Module(encoder_blocks))  
      decoder=Decoder(nn.Module(decoder_blocks))
      #Create the projection layer
      projection_layer=ProjectionLayer(d_model,target_vocab_size)

      #Create the transformer
      transformer=Transformer(encoder,decoder,src_embed,target_embed,src_pos,target_pos,projection_layer)

      #Initialising the parameters
      for p in transformer.parameters():
            if p.dim()>1:
                  nn.init.xavier_uniform_(p)

      return transformer



      
      

      
            

      







       
        
        
              
               


              
              
              