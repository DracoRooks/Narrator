# External Libs
import torch

# Custom Libs
from .globals import HyprParams
from .model import Model
from .splitDataset import trainData, valData, testData

# Loading octagrams
@torch.no_grad()
def loadBatch(split: str = 'train') -> tuple[torch.Tensor, torch.Tensor]:
    batchData = trainData if split == 'train' else valData if split == 'val' else testData
    ix = torch.randint(0, batchData.shape[0] - HyprParams.nBlock, (HyprParams.nBatch,))

    xb = torch.stack([batchData[i : i + HyprParams.nBlock] for i in ix])
    yb = torch.stack([batchData[i + 1 : i + HyprParams.nBlock + 1] for i in ix])

    return xb, yb

# Estimating Evaluation Loss
@torch.no_grad()
def estimateLoss(model: Model):
    lossEstimates = []
    for split in ['train', 'eval']:
        model.train() if split == 'train' else model.eval() if split == 'eval' else None
        lossi = torch.zeros(HyprParams.evalIters)
        for i in range(HyprParams.evalIters):
            x, y = loadBatch(split)
            logits, loss = model(x, y)
            lossi[i] = loss
        lossEstimates.append(lossi.mean(dim = 0).item())
    model.train()
    return f"Train Loss Estimate: {lossEstimates[0]:.4f}, Val Loss Estimate: {lossEstimates[1]:.4f}"
