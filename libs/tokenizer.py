# Custom Libs
from .loadDataset import file
from .globals import HyprParams

# Counting the number of unique characters (tokens)
uniqueChars = sorted(list(set(file)))
nUniqueChars = len(uniqueChars)

# Assigning unique identifiers for unique characters
stoi = {ch:idx for idx, ch in enumerate(uniqueChars)}
itos = {idx:ch for ch, idx in stoi.items()}

# Encoder / decoder
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[idx] for idx in l])

# Setting the nVocab
HyprParams.nVocab = nUniqueChars
