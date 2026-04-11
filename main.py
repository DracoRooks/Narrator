# Importing External Libs
import torch
from torch.optim import AdamW, lr_scheduler

# Custom Libs
from libs.tokenizer import encode, decode
from libs.globals import HyprParams
from libs.model import Model
from libs.utils import loadBatch, estimateLoss

# Order in the bleak world ruled by Chaos
gen = torch.manual_seed(12345)

model = Model().to(HyprParams.device)

optimizer = AdamW(model.parameters(), HyprParams.lr)
lrScheduler = lr_scheduler.StepLR(optimizer, HyprParams.lrDecayStep, HyprParams.lrDecay)

print("Training Started!")
for _ in range(HyprParams.nEpochs):

    x, y = loadBatch()

    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    lrScheduler.step()
    if _ % 800 == 0:
        estimateLoss(model)
print("Training Done!")

print("Saving Trained Parameter Configuration...")
torch.save(model.state_dict(), "./datasets/ri_gpt_v0.1.1.pth")
print("Trained Parameter Configuration Saved!")

print("Begining Inference...")
inputStr = 'Fang Yuan shouted with passion as he threw out a Gu Worm and yelled, "Liquor Worm Immortal Gu, Go!"'
out = model.generate(torch.tensor(encode(inputStr)).view(1, -1), 1000)
print(f"input: {decode(torch.tensor(encode(inputStr)).tolist())}")
print(f"output: {decode(out[0].tolist()[len(inputStr):])}")
