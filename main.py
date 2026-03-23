# Importing External Libs
import torch
from torch.nn import functional as func
from torch.optim import Adam, lr_scheduler

# Reading from file
fileBuffer = open("./datasets/ri.txt", "r", encoding = "utf-8")
file = fileBuffer.read().splitlines()
fileBuffer.close()

# Counting the number of unique characters (tokens) + 1 special token
uniqueChars = sorted(list(set(''.join(file))))
nUniqueChars = len(uniqueChars) + 1

# Assigning unique identifiers for unique characters
stoi = {ch:idx + 1 for idx, ch in enumerate(uniqueChars)}
stoi['<S>'] = 0
itos = {idx:ch for ch, idx in stoi.items()}

# Encoder / decoder
encode = lambda ch: stoi[ch]
decode = lambda idx: itos[idx]

# Loading 32grams
blockSize = 31
xs = []
ys = []
for line in file[:10000]:
    context = ['<S>'] * blockSize
    string = list(line) + ['<S>']
    for ch2 in string:
        xs.append([encode(ch1) for ch1 in context])
        ys.append(encode(ch2))
        context = context[1:] + [ch2]

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# Dividing the dataset
train = int(xs.shape[0] * 0.8)
dev = int(xs.shape[0] * 0.9)

Xtr = xs[:train]
Ytr = ys[:train]
print(f"Xtr: {Xtr.shape}, Ytr: {Ytr.shape}")

Xdev = xs[train:dev]
Ydev = ys[train:dev]
print(f"Xdev: {Xdev.shape}, Ydev: {Ydev.shape}")

Xtst = xs[dev:]
Ytst = ys[dev:]
print(f"Xtst: {Xtst.shape}, Ytst: {Ytst.shape}")

# Hyper-parameters
gen = torch.Generator().manual_seed(12345)
nNeurons = (64, 64, nUniqueChars)
nEmbedding = 16
nBatch = 409
lr = 0.01
lrDecay = 0.98
lrDecayStep = 100
epochs = 2000
momentum = nBatch / Xtr.shape[0]
tanhGain = 5/3
kaimin_init_w1 = tanhGain / ((blockSize * nEmbedding) ** 0.5)
kaimin_init_w2 = tanhGain / (nNeurons[0] ** 0.5)
eps = 1e-9

# Initialising model parameters
embed = torch.randn((nUniqueChars, nEmbedding), generator = gen, dtype = torch.float32) # Vector Embeddings (encodings) for each unique character

w1 = torch.randn((blockSize * nEmbedding, nNeurons[0]), generator = gen, dtype = torch.float32) * kaimin_init_w1
bngain1 = torch.ones((1, nNeurons[0]), dtype = torch.float32)
bnbias1 = torch.zeros((1, nNeurons[0]), dtype = torch.float32)

w2 = torch.randn((nNeurons[0], nNeurons[1]), generator = gen, dtype = torch.float32) * kaimin_init_w2
bngain2 = torch.ones((1, nNeurons[1]), dtype = torch.float32)
bnbias2 = torch.zeros((1, nNeurons[1]), dtype = torch.float32)

w3 = torch.randn((nNeurons[1], nNeurons[2]), generator = gen, dtype = torch.float32) * 0.001
b3 = torch.zeros((nNeurons[2],), dtype = torch.float32)

# Running params
bnmean1_running = torch.zeros(1, nNeurons[0], dtype = torch.float32)
bnstd1_running = torch.ones(1, nNeurons[0], dtype = torch.float32)
bnmean2_running = torch.zeros(1, nNeurons[1], dtype = torch.float32)
bnstd2_running = torch.ones(1, nNeurons[1], dtype = torch.float32)

freeParams: list[torch.Tensor] = [embed, w1, bngain1, bnbias1, w2, bngain2, bnbias2, w3, b3]

nFreeParams = 0
for param in freeParams:
    n = 1
    for dim in param.shape:
        n *= dim
    nFreeParams += n
print(f"Total Free Parameters: {nFreeParams}")

torch.no_grad()

# Training the model
for _ in range(epochs):

    # Creating a random mini-batch
    ix = torch.randint(0, Xtr.shape[0], (nBatch,), generator = gen)

    # Forward Pass
    emb = embed[Xtr[ix]]
    x = emb.view((-1, blockSize * nEmbedding))
    y = Ytr[ix]

    # Operations of 1st hidden layer
    h1prebn = x @ w1
    bnmean1 = h1prebn.mean(dim = 0, keepdim = True)
    bnvar1 = h1prebn.var(dim = 0, keepdim = True, unbiased = True)
    bnstd1 = (bnvar1 + eps) ** 0.5
    bnraw1 = (h1prebn - bnmean1) / bnstd1
    h1preact = bngain1 * bnraw1 + bnbias1
    h1 = h1preact.tanh()

    # Operations of 2st hidden layer
    h2prebn = h1 @ w2
    bnmean2 = h2prebn.mean(dim = 0, keepdim = True)
    bnvar2 = h2prebn.var(dim = 0, keepdim = True, unbiased = True)
    bnstd2 = (bnvar2 + eps) ** 0.5
    bnraw2 = (h2prebn - bnmean2) / bnstd2
    h2preact = bngain2 * bnraw2 + bnbias2
    h2 = h2preact.tanh()

    # Running approximations
    bnmean1_running = (1 - momentum) * bnmean1_running + momentum * bnmean1
    bnstd1_running = (1 - momentum) * bnstd1_running + momentum * bnstd1

    bnmean2_running = (1 - momentum) * bnmean2_running + momentum * bnmean2
    bnstd2_running = (1 - momentum) * bnstd2_running + momentum * bnstd2

    # Operations of output layer
    logits: torch.Tensor = (h2 @ w3) + b3
    loss = func.cross_entropy(logits, y)

    # Backward Pass
    dlogits = func.softmax(logits, dim = 1)
    dlogits[range(nBatch), y] -= 1
    dlogits /= nBatch

    dw3 = h2.T @ dlogits
    db3 = dlogits.sum(dim = 0)
    dh2 = dlogits @ w3.T

    dh2preact = (1.0 - (h2 ** 2)) * dh2

    dbngain2 = (bnraw2 * dh2preact).sum(dim = 0, keepdim = True)
    dbnbias2 = dh2preact.sum(dim = 0, keepdim = True)

    dh2prebn = bngain2 * ((bnstd2 * nBatch) ** -1) * (nBatch * dh2preact - dh2preact.sum(0) - (nBatch / (nBatch - 1)) * bnraw2 * ((dh2preact * bnraw2).sum(0)))

    dw2 = h1.T @ dh2prebn
    dh1 = dh2prebn @ w2.T

    dh1preact = (1.0 - (h1 ** 2)) * dh1

    dbngain1 = (bnraw1 * dh1preact).sum(dim = 0, keepdim = True)
    dbnbias1 = dh1preact.sum(dim = 0, keepdim = True)

    dh1prebn = bngain1 * ((bnstd1 * nBatch) ** -1) * (nBatch * dh1preact - dh1preact.sum(0) - (nBatch / (nBatch - 1)) * bnraw1 * ((dh1preact * bnraw1).sum(0)))

    dw1 = x.T @ dh1prebn
    dx = dh1prebn @ w1.T

    demb = dx.view(emb.shape)

    dembed = torch.zeros_like(embed)
    for i in range(Xtr[ix].shape[0]):
        for j in range(Xtr[ix].shape[1]):
            idx = Xtr[ix][i, j]
            dembed[idx] += demb[i, j]

    # UPDATE GRADIENTS
    embed -= lr * dembed
    w1 -= lr * dw1
    bngain1 -= lr * dbngain1
    bnbias1 -= lr * dbnbias1
    w2 -= lr * dw2
    bngain2 -= lr * dbngain2
    bnbias2 -= lr * dbnbias2
    w3 -= lr * dw3
    b3 -= lr * db3

    if _ % lrDecayStep == 0: lr *= lrDecay
    if _ % 20 == 0: print(f"Training Loss: {loss.item()}")


print("Training Done!")
print(f"Total Free Parameters: {nFreeParams}")

params = {
    "uniqueChars": uniqueChars,
    "hyperParams": {
        "gen": gen,
        "blockSize": blockSize,
        "nNeurons": nNeurons,
        "nEmbedding": nEmbedding,
        "nBatch": nBatch,
        "lr": {
            "rate": lr,
            "decay": lrDecay,
            "step": lrDecayStep
        },
        "epochs": epochs,
        "momentum": momentum,
        "tanhGain": tanhGain,
        "kaimin_init_w1": kaimin_init_w1,
        "kaimin_init_w2": kaimin_init_w2
    },
    "batchNormParams": {
        "bnmean1_running": bnmean1_running,
        "bnstd1_running": bnstd1_running,
        "bnmean2_running": bnmean2_running,
        "bnstd2_running": bnstd2_running,
    },
    "freeParams": freeParams
}

torch.save({"params": params}, "./datasets/32gram_manual_backprop.pth")
print("Trained weight configuration saved!")

# Validation Results
ix = torch.randint(0, Xdev.shape[0], (nBatch,), generator = gen)
x = embed[Xdev[ix]].view(-1, blockSize * nEmbedding)
y = Ydev[ix]

# Atomic operations of 1st hidden layer
h1prebn = x @ w1
bnraw1 = (h1prebn - bnmean1_running) / bnstd1_running
h1preact = bngain1 * bnraw1 + bnbias1
h1 = h1preact.tanh()

# Atomic operations of 2st hidden layer
h2prebn = h1 @ w2
bnraw2 = (h2prebn - bnmean2_running) / bnstd2_running
h2preact = bngain2 * bnraw2 + bnbias2
h2 = h2preact.tanh()

logits = (h2 @ w3) + b3
loss = func.cross_entropy(logits, y)
print(f"Validation Loss: {loss.item()}")
