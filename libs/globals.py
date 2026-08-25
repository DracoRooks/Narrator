# Internal Libs
from dataclasses import dataclass

# External Libs
import torch

@dataclass
class HyprParams:
    # Model Hyper Parameters
    nBlock = 16
    nBatch = 256
    nEmbed = 160
    nLayer = 4
    nHeads = 8
    pDrop = 0.1
    pWeightDecay = 0.012

    # Training Hyper Parameters
    lr = 2e-4
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
    nVocab = 300 # Including the 256 byte level tokens

    # Inference Hyper Params
    topk = 100
