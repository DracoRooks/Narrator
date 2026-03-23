# Importing External Libs
import torch
from torch.nn import functional as func

# Loading The Pre-trained Octagram Language Model
modelToSample = "./datasets/32gram_param_configuration_optimized.pth"
state = torch.load(modelToSample, weights_only = False)
params = state["params"]

# Loading params from the saved state
uniqueChars = params["uniqueChars"]
hyperParams = params["hyperParams"]
batchNormParams = params["batchNormParams"]
freeParams = params["freeParams"]

# Reading all freeParams from the saved file
embed = freeParams["embedding"]

w1 = freeParams["w1"]
bngain1 = freeParams["bngain1"]
bnbias1 = freeParams["bnbias1"]

w2 = freeParams["w2"]
bngain2 = freeParams["bngain2"]
bnbias2 = freeParams["bnbias2"]

w3 = freeParams["w3"]
b3 = freeParams["b3"]

# Reading Batch Normalisation Parameters
bnmean1_running = batchNormParams["bnmean1_running"]
bnstd1_running = batchNormParams["bnstd1_running"]

bnmean2_running = batchNormParams["bnmean2_running"]
bnstd2_running = batchNormParams["bnstd2_running"]

# Reading the rng from hyperParams
gen = hyperParams["gen"]

# Assigning unique identifiers for unique characters
stoi = {ch:idx + 1 for idx, ch in enumerate(uniqueChars)}
stoi['<S>'] = 0
itos = {idx:ch for ch, idx in stoi.items()}

# Encoder / decoder
encode = lambda ch: stoi[ch]
decode = lambda idx: itos[idx]

# Sampling
result = []
for _ in range(20):
    line = ""
    context = [0] * hyperParams["blockSize"]
    while True:
        x = embed[torch.tensor(context)].view(1, -1)

        h1preact = x @ w1
        h1preact = (h1preact - bnmean1_running) / bnstd1_running
        h1preact = bngain1 * h1preact + bnbias1
        h1 = h1preact.tanh()

        h2preact = h1 @ w2
        h2preact = (h2preact - bnmean2_running) / bnstd2_running
        h2preact = bngain2 * h2preact + bnbias2
        h2 = h2preact.tanh()

        logits = (h2 @ w3) + b3
        probs = func.softmax(logits, dim = 1)

        y = torch.multinomial(probs, 1, True, generator = gen).item()

        if y == 0: break

        line += decode(y)
        context = context[1:] + [y]
    result += [line]

print('\n'.join(result))
