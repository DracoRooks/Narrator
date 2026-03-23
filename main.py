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

# Loading octagrams
blockSize = 7
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
nNeurons = (64, nUniqueChars)
nEmbedding = 16
nBatch = 4096
lr = 0.1
lrDecay = 0.96
lrDecayStep = 100
epochs = 100000

# Initialising model parameters
embed = torch.randn((nUniqueChars, nEmbedding), generator = gen, dtype = torch.float32) # Vector Embeddings (encodings) for each unique character

w1 = torch.randn((blockSize * nEmbedding, nNeurons[0]), generator = gen, dtype = torch.float32) * 0.001
b1 = torch.zeros((nNeurons[0],), dtype = torch.float32)

w2 = torch.randn((nNeurons[0], nNeurons[1]), generator = gen, dtype = torch.float32) * 0.001
b2 = torch.zeros((nNeurons[1],), dtype = torch.float32)

freeParams = {
    "embedding": embed,
    "w1": w1,
    "b1": b1,
    "w2": w2,
    "b2": b2
}

nFreeParams = 0
for param in freeParams.values():
    n = 1
    for dim in param.shape:
        n *= dim
    nFreeParams += n
print(f"Total Free Parameters: {nFreeParams}")

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

    h = ((x @ w1) + b1).tanh()
    logits = (h @ w2) + b2
    loss = func.cross_entropy(logits, y)

    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lrScheduler.step()

    if _ % 10000 == 0: print(f"Training Loss: {loss.item()}")

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
        "epochs": epochs
    },
    "freeParams": freeParams
}

torch.save({"params": params, "optim": optimizer.state_dict}, "./datasets/octagram_param_configuration.pth")
print("Trained weight configuration saved!")

# Validation Results
ix = torch.randint(0, Xdev.shape[0], (nBatch,), generator = gen)
x = embed[Xdev[ix]].view(-1, blockSize * nEmbedding)
y = Ydev[ix]
h = ((x @ w1) + b1).tanh()
logits = (h @ w2) + b2
loss = func.cross_entropy(logits, y)
print(f"Validation Loss: {loss.item()}")
