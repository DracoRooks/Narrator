# Internal Libs
from dataclasses import dataclass

# External Libs
import torch

@dataclass
class HyprParams:
    # Model Hyper Parameters
    nBlock = 32
    nBatch = 1024
    nEmbed = 48
    nLayer = 3
    nHeads = 6
    nDrop = 0.2
    lr = 1e-4
    lrDecay = 0.975
    lrDecayStep = 80
    nEpochs = 4001

    # Compute Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loss Estimation Iterations
    evalIters = 10

    # Vocab size set by tokenizer
    nVocab = -1
