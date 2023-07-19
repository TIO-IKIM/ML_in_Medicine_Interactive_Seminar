# Author: Prometheus9920

# An additional solution by the tutors, based on the tutorial in the VisionTransformer.ipynb notebook is available in Course_Materials/tutor_solutions/vit_tutor.py .

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedPositionalEmbeddings(nn.Module):

    """
    embed_dim refers to the size of the embedding dimension
    patches refers to the patch size
    """

    def __init__(self, embed_dim: int, patches: int):
        
        super(LearnedPositionalEmbeddings,self).__init__()

        self.positional_encodings = nn.Parameter(torch.zeros(1, patches+1, embed_dim), requires_grad=True)
        # This creates a tensor of dimension (1, patch size+1, embedding dimension).
        # All parameters are learnable and zero-initialized.

    def forward(self, x: torch.Tensor):

        pe = self.positional_encodings[:x.shape[0]] # For the same size along the batch dimension

        return x + pe
    
class Transformer(nn.Module):

    """
    Single transformer encoder block.
    Dimensions of the forwarded tensor are commented in the forward method.
    """

    def __init__(self,input_dim,embed_dim,heads,patches):
        # input_dim = P^2, embed_dim=D,h eads=h, patches=N=H*W/P^2
        
        super(Transformer, self).__init__()
        
        self.heads=heads            
        self.embed_dim=embed_dim
        self.input_dim=input_dim
        self.patches=patches
        self.head_dim=int(self.embed_dim/self.heads) # embed_dim/heads = head_size

        # Initial normalization
        self.norm1=nn.LayerNorm([self.patches,self.embed_dim])

        # Linear layer to compute Queries, Keys, and Values for each head h
        self.linear1=nn.Linear(self.embed_dim,self.embed_dim*3,bias=False)#y=A*x

        # Second normalization
        self.norm2=nn.LayerNorm([self.patches,self.embed_dim])

        # MLP after self-attention
        self.lin = nn.Sequential(nn.Linear(self.embed_dim,self.embed_dim*3,bias=True),
                                    nn.GELU(),
                                    nn.Linear(self.embed_dim*3,self.embed_dim,bias=True),
                                    nn.GELU())
        
 
    def forward(self, x): #input (B,N+1,D)
        
        batch_size=x.shape[0]
        
        # Norm 1
        out=self.norm1(x)

        # Create Queries, Keys, Values
        out = self.linear1(out) # Batch-Größe, N patches+1, embed dim D*3
        
        # Reshape
        out= out.reshape(batch_size,self.patches,3,self.heads,self.head_dim)#Batch-Größe, N patches +1,3 (Q K V), h heads, head dim d_k
        
        # Permute
        out=out.permute(2,0,3,1,4) # (3 (Q K V), Batch size, h heads, N patches +1, head dim d_k)
        q,k,v=out[0],out[1],out[2] # Query, Key, Value

        # Transpose Keys
        k_t=k.transpose(-2,-1) # (Batch size, h heads, head dim d_k, N patches (+1))

        # Matrix multiplication of Queries, Keys; sqrt(head dim d_k)
        dp=(q@k_t)/torch.sqrt(torch.tensor(self.head_dim)) # Batch size, h heads, N patches +1, head dim d_k
        dp=dp.softmax(-2) # use softmax line-by-line

        # Multiply with values
        weighted_dp=dp@v # (Batch size, h heads, head dim d_k, N patches (+1))

        # Transpose
        weighted_dp=weighted_dp.transpose(1,2) # (Batch size, N patches (+1), h heads, head dim d_k)

        # Concat h heads and head dim d_k back together
        weighted_dp=weighted_dp.flatten(2) # (Batch size, N patches (+1), embed dim D)

        out=x+weighted_dp # add skip connection
        out1=out # save for the next skip connection

        # Norm 2
        out=self.norm2(out)

        # MLP
        out=self.lin(out)   

        return out + out1 # return result and skip connection
    
class ViT(nn.Module):

    """
    Complete Vision Transformer, assembled from Transformer blocks.
    """

    def __init__(self,input_dim,classes,embed_dim,heads,patches):
        #input_dim = P^2, classes, embed_dim=D, heads=h, patches=N=H*W/P^2

        super(ViT, self).__init__()
        
        self.input_dim=input_dim
        self.heads=heads            
        self.embed_dim=embed_dim
        self.head_dim=self.embed_dim/self.heads # embed_dim/heads = head_size
        self.input_dim=input_dim
        self.classes=classes
        self.patches=patches

        # Calculate what size patches to split images into
        self.patch_size = int(np.sqrt(self.input_dim))

        # Positional embedding
        self.pos=LearnedPositionalEmbeddings(self.embed_dim,self.patches)

        # Linear embedding
        self.lin_emb=nn.Linear(self.input_dim,self.embed_dim,bias=False) # Change patch_size to embedding dimension D

        # CLS Token
        self.CLS_Token = nn.Parameter(torch.zeros(self.embed_dim), requires_grad=True)

        #Transformer
        self.trans1=Transformer(input_dim=self.input_dim,embed_dim=self.embed_dim,heads=self.heads,patches=self.patches+1)
        self.trans2=Transformer(input_dim=self.input_dim,embed_dim=self.embed_dim,heads=self.heads,patches=self.patches+1)
        self.trans3=Transformer(input_dim=self.input_dim,embed_dim=self.embed_dim,heads=self.heads,patches=self.patches+1)
        self.trans4=Transformer(input_dim=self.input_dim,embed_dim=self.embed_dim,heads=self.heads,patches=self.patches+1)

        #CLS MLP
        self.norm=nn.LayerNorm([self.embed_dim])
        self.cls_MLP = nn.Sequential(nn.Linear(self.embed_dim,self.embed_dim*3,bias=True),
                                    nn.Tanh(),
                                    nn.Linear(self.embed_dim*3,self.embed_dim,bias=True),
                                    nn.Tanh(),
                                    nn.Linear(self.embed_dim,3,bias=True),
                                    nn.Tanh())    
 
    def forward(self, x):
        # input (B,N,input_dim)

        batch_size=x.shape[0]

        # Change data into correct shape => (B,C,H,W) becomes (B,N,C*P^2)
        x=x.unfold(2,self.patch_size,self.patch_size).unfold(3,self.patch_size,self.patch_size).transpose(1,3).transpose(1,2).flatten(1,2)
        x=x.flatten(2,4)

        # Linear embedding
        out = self.lin_emb(x)

        # CLS tokens 
        clsToken=self.CLS_Token.repeat(batch_size,1)
        # Add one dimension
        clsToken=torch.unsqueeze(clsToken, dim=1)
        # Concat the CLS tokens to the patches
        out=torch.cat((clsToken,out),dim=1)
        
        # Positional embedding
        out = self.pos(out)

        # Transformer layers
        out = self.trans1(out) #Transformer
        out = self.trans2(out) #Transformer
        out = self.trans3(out) #Transformer
        out = self.trans4(out) #Transformer

        # Keep only the CLS tokens
        out=out[:,0]

        # Classifier MLP based on CLS tokens
        out=self.norm(out)
        out=self.cls_MLP(out)

        return out 

if __name__ == "__main__":

    # Build the model and count its parameters.
    model = ViT(input_dim=256, classes=3, embed_dim=128, heads=8, patches=256)
    numel_list = [p.numel() for p in model.parameters()]
    print(sum(numel_list))