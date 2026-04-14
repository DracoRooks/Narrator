# Custom Libs
from .loadDataset import file
from .tokenizer_cpp import tokenizer
from .globals import HyprParams

# External Libs
import torch

# Encoding the dataset
data = torch.tensor(tokenizer.encode(file), dtype = torch.long, device = HyprParams.device)

# Setting the size of train/val/test splits
nTrain = int(0.8 * data.shape[0])
nVal = int(0.9 * data.shape[0])

# Splitting the dataset
trainData = data[:nTrain]
valData = data[nTrain:nVal]
testData = data[nVal:]
