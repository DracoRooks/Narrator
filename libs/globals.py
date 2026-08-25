# Internal Libs
from dataclasses import dataclass

# External Libs
import torch

@dataclass
class HyprParams:
    # Model Hyper Parameters
    nBatch = 24
    nBlock = 512
    nEmbed = 288
    nLayer = 4
    nHeads = 12
    pDrop = 0.1
    pWeightDecay = 0.01

    # Training Hyper Parameters
    lr = 8e-4
    warmup = 0.1
    initDivFactor = 20
    termDivFactor = 2e4
    nEpochs = 801

    # Compute Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loss Estimation Params
    evalIters = 20
    pEval = 0.1 # evaluate the model after what fraction of nEpochs

    # Vocab size set by tokenizer
    nVocab = 1256 # Including the 256 byte level tokens

    # Inference Hyper Params
    topk = 100
