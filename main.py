# Importing External Libs
import torch
import torch.nn.functional as func

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

# Loading bigrams
xs = []
ys = []
for line in file[:30000]:
    string = ['<S>'] + list(line) + ['<S>']
    for ch1, ch2 in zip(string, string[1:]):
        xs.append(encode(ch1))
        ys.append(encode(ch2))

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xenc = func.one_hot(xs, num_classes = nUniqueChars).float()

# Model parameters
nInputs = nUniqueChars
nNeurons = (nUniqueChars,)
gen = torch.Generator().manual_seed(1234)
weights = torch.randn((nInputs, nNeurons[0]), generator = gen, requires_grad = True)
zeroGrad = torch.zeros(weights.shape)
lr = 1
# Training Model
for i in range(200):

    # Forward Pass
    logits = xenc @ weights
    loss = func.cross_entropy(logits, ys)

    # Backward Pass
    weights.grad = zeroGrad
    loss.backward()

    # Updating Model Params
    if i < 20: lr = 10
    elif i < 50: lr = 1
    elif i < 100: lr = 0.5
    elif i < 150: lr = 0.1
    else: lr = 0.01
    weights.data -= lr * weights.grad

    print(f"{i}th loss: {loss.item()} with lr: {lr}")

print("Training Done!")

# Sampling from the trained model
result = []
ix = 0

for _ in range(50):
    line = ""
    while True:
        logits = func.one_hot(torch.tensor([ix]), num_classes = nUniqueChars).float() @ weights
        counts = logits.exp()
        probs = counts / counts.sum(dim = 1, keepdim = True)

        ix = torch.multinomial(probs, replacement = True, num_samples = 1).item()
        if ix == 0: break
        line += decode(ix)
    result.append(line)

print('\n'.join(result))
