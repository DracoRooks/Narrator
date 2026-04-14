# Custom Libs
from .loadDataset import file
from .globals import HyprParams

filebuffer = open("./datasets/ri.txt")
file = filebuffer.read()
filebuffer.close()

def getCodePoints(data: str):
    codePoints = list(map(int, data.encode("utf-8")))
    return codePoints

def getPairs(codePoints: list[int]):
    pairs: dict[tuple[int, int], int] = {}
    for pair in zip(codePoints, codePoints[1:]):
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs

def doMerge(tokens: list[int], pair: tuple[int, int], idx: int):
    newTokens: list[int] = []
    i: int = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            newTokens.append(idx)
            i += 2
        else:
            newTokens.append(tokens[i])
            i += 1
    return newTokens


mergeForest: dict[tuple[int, int], int] = {}
vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
def train(data: str, cycles: int):
    tokens = getCodePoints(data)

    i: int = 0

    while i < cycles:
        pairs = getPairs(tokens)
        mostFreqPair = max(pairs, key = lambda pair: pairs.get(pair, float()))
        tokens = doMerge(tokens, mostFreqPair, 256 + i)
        mergeForest[mostFreqPair] = 256 + i
        print(f"Updated Total Tokens: {len(tokens)}, Merging {mostFreqPair} into {256 + i}")
        i += 1

    for (p0, p1), idx in mergeForest.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    return tokens, mergeForest, vocab

def encode(string: str):
    tokens = getCodePoints(string)
    while True:
        pairs = getPairs(tokens)
        minPair = min(pairs, key = lambda pair: mergeForest.get(pair, float("inf")))
        if minPair not in mergeForest: break
        tokens = doMerge(tokens, minPair, mergeForest[minPair])
    return tokens

def decode(tokens: list[int]):
    utf8 = b"".join(vocab[idx] for idx in tokens)
    text = utf8.decode("utf-8", "replace")
    return text

train(file, 2)

# Setting the nVocab
HyprParams.nVocab = len(vocab)
