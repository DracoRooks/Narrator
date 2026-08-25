# Custom Libs
from .globals import HyprParams

# External Libs
import torch
from Scribe import BytePairEncoder

# Training the tokenizer
print("Training Tokenizer...")
BPE = BytePairEncoder()
BPE.train("./datasets/ri.txt", HyprParams.nVocab - 256, False, True)
print("Training Tokenizer... Done!")

# Fetching the encoded (according to the tokenizer) dataset
print("Encoding Data...")
dataInTokens = []
with open("./datasets/ri.txt", "r", encoding="utf-8") as buffer:
    dataInString = buffer.read()
    dataInTokens = BPE.encode(dataInString)
print("Encoding Data... Done!")

data = torch.tensor(dataInTokens, dtype = torch.long)

# Setting the size of train/val/test splits
nTrain = int(0.8 * data.shape[0])
nVal = int(0.9 * data.shape[0])

# Splitting the dataset
trainData = data[:nTrain]
valData = data[nTrain:nVal]
testData = data[nVal:]
