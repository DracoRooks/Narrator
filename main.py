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
for line in file:
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
nBatch = 8196
lr = 0.01
lrDecay = 0.99
lrDecayStep = 100
epochs = 100000
momentum = nBatch / Xtr.shape[0]
tanhGain = 5/3
kaimin_init_w1 = tanhGain / ((blockSize * nEmbedding) ** 0.5)
kaimin_init_w2 = tanhGain / (nNeurons[0] ** 0.5)

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

freeParams = {
    "embedding": embed,
    "w1": w1,
    "bngain1": bngain1,
    "bnbias1": bnbias1,
    "w2": w2,
    "bngain2": bngain2,
    "bnbias2": bnbias2,
    "w3": w3,
    "b3": b3
}

nFreeParams = 0
for param in freeParams.values():
    n = 1
    for dim in param.shape:
        n *= dim
    nFreeParams += n

for param in freeParams.values():
    param.requires_grad_() # Make sure PyTorch is recording the gradient graph of all the model's parameters

optimizer = Adam(freeParams.values(), lr)
lrScheduler = lr_scheduler.StepLR(optimizer, lrDecayStep, lrDecay)

# Training the model
for _ in range(epochs):

    # Creating a random mini-batch
    ix = torch.randint(0, Xtr.shape[0], (nBatch,), generator = gen)

    # Forward Pass
    x = embed[Xtr[ix]].view((-1, blockSize * nEmbedding))
    y = Ytr[ix]

    h1preact = x @ w1
    bnmean1 = h1preact.mean(dim = 0, keepdim = True)
    bnstd1 = h1preact.std(dim = 0, keepdim = True)
    h1preact = (h1preact - bnmean1) / bnstd1
    h1preact = bngain1 * h1preact + bnbias1
    h1 = h1preact.tanh()

    h2preact = h1 @ w2
    bnmean2 = h2preact.mean(dim = 0, keepdim = True)
    bnstd2 = h2preact.std(dim = 0, keepdim = True)
    h2preact = (h2preact - bnmean2) / bnstd2
    h2preact = bngain2 * h2preact + bnbias2
    h2 = h2preact.tanh()

    with torch.no_grad():
        bnmean1_running = (1 - momentum) * bnmean1_running + momentum * bnmean1
        bnstd1_running = (1 - momentum) * bnstd1_running + momentum * bnstd1

        bnmean2_running = (1 - momentum) * bnmean2_running + momentum * bnmean2
        bnstd2_running = (1 - momentum) * bnstd2_running + momentum * bnstd2

    logits = (h2 @ w3) + b3
    loss = func.cross_entropy(logits, y)

    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lrScheduler.step()

    if _ % 1 == 0: print(f"Training Loss: {loss.item()}")

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

torch.save({"params": params, "optim": optimizer.state_dict}, "./datasets/octagram_manual_backprop.pth")
print("Trained weight configuration saved!")

# Validation Results
ix = torch.randint(0, Xdev.shape[0], (nBatch,), generator = gen)
x = embed[Xdev[ix]].view(-1, blockSize * nEmbedding)
y = Ydev[ix]

h1preact = x @ w1
h1preact = (h1preact - bnmean1_running) / bnstd1_running
h1preact = bngain1 * h1preact + bnbias1
h1 = h1preact.tanh()

h2preact = h1 @ w2
h2preact = (h2preact - bnmean2_running) / bnstd2_running
h2preact = bngain2 * h2preact + bnbias2
h2 = h2preact.tanh()

logits = (h2 @ w3) + b3
loss = func.cross_entropy(logits, y)
print(f"Validation Loss: {loss.item()}")
