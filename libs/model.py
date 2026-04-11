# External Libs
import torch
from torch import nn
from torch.nn import functional as nnfunc

# Custom Libs
from .globals import HyprParams

# Self-Attention
class SelfAttention(nn.Module):

    def __init__(self, sizeHead: int) -> None:
        super().__init__()

        self.key = nn.Linear(HyprParams.nEmbed, sizeHead, bias = False) # (T, C)
        self.query = nn.Linear(HyprParams.nEmbed, sizeHead, bias = False) # (T, C)
        self.value = nn.Linear(HyprParams.nEmbed, sizeHead, bias = False) # (T, C)
        self.register_buffer('tril', torch.tril(torch.ones((HyprParams.nBlock, HyprParams.nBlock)))) # (T, T)

        self.dropout = nn.Dropout(HyprParams.nDrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        keys = self.key(x) # (B, T, C)
        queries = self.query(x) # (B, T, C)

        affinities = queries @ keys.transpose(dim0 = -2, dim1 = -1) * C ** -0.5 # (B, T, C) @ (B, C, T) ---> (B, T, T)
        affinities = affinities.masked_fill(self.tril == 0, float('-inf')) # (B, T, T)
        affinities = nnfunc.softmax(affinities, dim = -1) # (B, T, T)

        affinities = self.dropout(affinities)

        values = self.value(x) # (B, T, C)

        out = affinities @ values # (B, T, T) @ (B, T, C) ---> (B, T, C)

        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, nHeads: int, sizeHead: int) -> None:
        super().__init__()

        self.heads = nn.ModuleList([SelfAttention(sizeHead) for _ in range(nHeads)])
        self.projection = nn.Linear(HyprParams.nEmbed, HyprParams.nEmbed)

        self.dropout = nn.Dropout(HyprParams.nDrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim = -1)
        out = self.projection(out)

        out = self.dropout(out)

        return out

# Post-Attention Feed Forward
class FeedForward(nn.Module):

    def __init__(self, fanIn: int, fanOut: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(fanIn, 8 * fanOut),
            nn.PReLU(init = 0.1),
            nn.Linear(8 * fanOut, fanOut),
            nn.Dropout(HyprParams.nDrop),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.net(x)

        return x

# Attention/Feed-Forward/Norm Block
class Block(nn.Module):

    def __init__(self, fanIn: int, fanOut: int) -> None:
        super().__init__()

        sizeHead = fanIn // HyprParams.nHeads

        self.selfAttention = MultiHeadAttention(HyprParams.nHeads, sizeHead)
        self.feedForward = FeedForward(fanIn, fanOut)

        self.layerNorm1 = nn.LayerNorm(HyprParams.nEmbed)
        self.layerNorm2 = nn.LayerNorm(HyprParams.nEmbed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.selfAttention(self.layerNorm1(x))
        x = x + self.feedForward(self.layerNorm2(x))

        return x

# Model
class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.tokenEmbeddingTable = nn.Embedding(HyprParams.nVocab, HyprParams.nEmbed)
        self.positionEmbeddingTable = nn.Embedding(HyprParams.nBlock, HyprParams.nEmbed)
        self.blocks = nn.Sequential(*[Block(HyprParams.nEmbed, HyprParams.nEmbed) for _ in range(HyprParams.nLayer)])
        self.layerNorm = nn.LayerNorm(HyprParams.nEmbed)

        self.linear = nn.Linear(HyprParams.nEmbed, HyprParams.nVocab)

    def forward(self, idx: torch.Tensor, targets = None):
        tokEmbed = self.tokenEmbeddingTable(idx) # (B, T, nEmbed)
        posEmbed = self.tokenEmbeddingTable(idx) # (B, T, nEmbed)
        x = tokEmbed + posEmbed # (B, T, nEmbed)

        preLogits = self.blocks(x) # (B, T, nUniqueChars)
        logits = self.layerNorm(preLogits)
        logits = self.linear(logits)

        if targets == None:
            return logits
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = nnfunc.cross_entropy(logits, targets)

            return logits, loss

    def generate(self, idx, length):
        for _ in range(length):
            idx_crop = idx[:, -HyprParams.nBlock:]
            logits = self(idx_crop)
            probs = nnfunc.softmax(logits[:, -1, :], dim = 1)
            nextIdx = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, nextIdx), dim = 1)

        return idx
