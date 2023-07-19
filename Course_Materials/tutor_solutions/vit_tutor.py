import torch, torch.nn as nn
import numpy as np
from typing import Union, Tuple, List

class TransformerEncoderBlock(torch.nn.Module):

    """
    Performs Scaled Dot-Product Self-Attention with residual connections,
    as described in the Encoder blocks of Dosovitskiy et al 2021.

    The block must know the sequence length and embedding size.
    The key dimension (called d_model in the paper) is inferred as embed_dim // n_heads .
    Additionally, the block can be given any number of heads - be careful when choosing n_heads, as the number of
    heads is multiplied by key_dim to make the final width
    """

    def __init__(self, seq_length: int, emb_length: int, n_heads: int):
        
        super(TransformerEncoderBlock, self).__init__()

        self.seq_length = seq_length
        self.emb_length = emb_length
        self.n_heads = n_heads

        assert self.emb_length % self.n_heads == 0
        self.key_dim = emb_length // n_heads

        self.afn = nn.ReLU()
        self.fc1 = nn.Linear(self.emb_length, self.emb_length)
        self.fc2 = nn.Linear(self.emb_length, self.emb_length)
        self.norm1 = nn.LayerNorm(self.emb_length)
        self.norm2 = nn.LayerNorm(self.emb_length)
        self.gamma1 = nn.Parameter(torch.tensor([1.])) # Strength of attention outputs compared to residual connection
        self.gamma2 = nn.Parameter(torch.tensor([1.]))
        self.qkv_proj = nn.Linear(self.emb_length, 3 * self.emb_length, bias = False)
        
    def forward(self, x: torch.Tensor):

        ## Generate Q, K, and V, and split the tensor across the number of heads
        x = self.norm1(x)
        skip = x
        qkv = self.qkv_proj(x)
        batch_size = x.size()[0]
        qkv = qkv.reshape(batch_size, self.seq_length + 1, self.n_heads, 3 * self.key_dim) # seq_length + 1 from cls token
        qkv = qkv.permute(0, 2, 1, 3) # -> [Batch, NHead, SeqLen, KDim]
        q, k, v = qkv.chunk(3, dim=-1)

        ## Perform Multi-Head Self-Attention
        # The automated parallelizing of large matrice multiplications does the actual optimization for us
        scale = (1 / np.sqrt(self.key_dim))
        attn_map = nn.functional.softmax(scale * (q @ k.mT), dim=1)
        x = attn_map @ v

        ## Get back original shape
        x = x.permute(0, 2, 1, 3) # -> [Batch, SeqLen, NHead, KDim]
        x = x.reshape(batch_size, self.seq_length + 1, self.emb_length) # seq_length + 1 from cls token

        ## Linear Block I
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.afn(x)

        ## Residual Connection and LayerNorm I
        x = x + self.gamma1 * skip
        skip = x

        ## Linear Block II
        x = self.fc2(x)
        x = self.afn(x)

        ## Residual Connection and LayerNorm II
        x = x + self.gamma2 * skip

        return x

class VisionTransformer(torch.nn.Module):
    """
    A Vision Transformer in the style of Dosovitskiy et al. 2021.

    num_classes - Number of classes in classification problem. Must be int.
    image_size - Must be a tuple of ints
    patch_size - Must be int. Patches are square.
    emb_length - Size of the patch embedding. Must be int.
    attn_blocks - Number of TransformerEncoder blocks before the final MLP head. Must be int.
    heads_per_block - Number of heads for MultiHeadAttention. Must be int or list of ints.
    - The dimensions of QKV are inferred from emb_length and n_heads.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: Tuple[int, int],
        patch_size: int = 16,
        emb_length: int = 512,
        attn_blocks: int = 6,
        heads_per_block: Union[List[int, ], int] = [4] * 6,
        ):

        super(VisionTransformer, self).__init__()

        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.emb_length = emb_length
        self.attn_blocks = attn_blocks
        self.heads_per_block = heads_per_block
        
        # Guarantee inputs are not garbage
        assert self.image_size[0] % self.patch_size == 0
        assert self.image_size[1] % self.patch_size == 0
        self.seq_length = int(self.image_size[0] * self.image_size[1] / self.patch_size**2)

        assert isinstance(self.attn_blocks, int)

        if isinstance(self.heads_per_block, int):
            self.heads_per_block = [self.heads_per_block]*self.attn_blocks
        else:
            assert isinstance(heads_per_block, list) and all(isinstance(h, int) for h in heads_per_block)

        # Build all layers
        self.embed_conv = nn.Conv2d(in_channels = 1, out_channels = self.emb_length, kernel_size = self.patch_size, stride = self.patch_size)
        self.embed_pos = nn.Parameter(torch.zeros((1, self.seq_length + 1, self.emb_length)))
        self.cls_token = nn.Parameter(torch.zeros((1, 1, self.emb_length)))
        self._build_attn_blocks()
        self.final_norm = nn.LayerNorm([1, self.emb_length])
        self.classifier = nn.Linear(self.emb_length, self.num_classes)

    def _build_attn_blocks(self):

        self.blocks = []
        for n in range(self.attn_blocks):
            self.blocks.append(f"AttnBlock_{n}")
            self.add_module(f"AttnBlock_{n}", TransformerEncoderBlock(
                seq_length = self.seq_length,
                emb_length = self.emb_length,
                n_heads = self.heads_per_block[n]
                ))
            pass

    def forward(self, x: torch.Tensor):
        # Initial embedding
        x = self.embed_conv(x)                     # -> B x ES x PS x PS

        # Permute dimensions so Sequence Length is in the channel dimension
        x = torch.permute(x.flatten(2), (0, 2, 1)) # -> B x SL x ES

        # Add CLS token (see Vaswani et al. 2017)
        batch_size = x.size()[0]
        x = torch.cat([self.cls_token.repeat((batch_size, 1, 1)), x], dim = 1)

        # Positional embedding
        x = x + self.embed_pos
        
        # Forward pass through all of the transformer's self-attention blocks
        for n, block in enumerate(self.blocks):
            x = getattr(self, block)(x)

        # Toss out all tokens in the sequence except the cls token
        x = x[:, 0, :].unsqueeze(1)

        # Final norm, flatten, and MLP head
        x = self.final_norm(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x