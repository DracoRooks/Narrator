# Importing External Libs
import torch
from torch.nn import functional as func

# Loading The Pre-trained Octagram Language Model
modelToSample = "./datasets/octagram_param_configuration.pth"
state = torch.load("./datasets/octagram_param_configuration.pth", weights_only = False)
params = state["params"]

# Loading params from the saved state
uniqueChars = params["uniqueChars"]
hyperParams = params["hyperParams"]
freeParams = params["freeParams"]

# Reading all freeParams from the saved file
embed = freeParams["embedding"]

w1 = freeParams["w1"]
b1 = freeParams["b1"]

w2 = freeParams["w2"]
b2 = freeParams["b2"]

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
    context = [0] * 7
    while True:
        x = embed[torch.tensor(context)].view(1, -1)

        h = ((x @ w1) + b1).tanh()
        logits = (h @ w2) + b2
        probs = func.softmax(logits, dim = 1)

        y = torch.multinomial(probs, 1, True, generator = gen).item()

        if y == 0: break

        line += decode(y)
        context = context[1:] + [y]
    result += [line]

print('\n'.join(result))
