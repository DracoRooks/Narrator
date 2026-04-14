# Importing External Libs
import torch
from torch.optim import AdamW, lr_scheduler

# Custom Libs
from libs.globals import HyprParams
from libs.tokenizer_cpp import tokenizer
from libs.model import Model
from libs.utils import loadBatch, estimateLoss

# Order in the bleak world ruled by Chaos
gen = torch.manual_seed(12345)

model = Model().to(HyprParams.device)

optimizer = AdamW(model.parameters(), HyprParams.lr, weight_decay = HyprParams.pWeightDecay)
lrScheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=HyprParams.lr,
    total_steps=HyprParams.nEpochs,
    pct_start=HyprParams.warmup,
    anneal_strategy='cos',
    div_factor=HyprParams.initDivFactor,
    final_div_factor=HyprParams.termDivFactor
)

print("Training Started!")
model.train()
for _ in range(HyprParams.nEpochs):

    x, y = loadBatch()

    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    lrScheduler.step()
    if _ % int(HyprParams.nEpochs * HyprParams.pEval) == 0:
        print(f"Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}, {estimateLoss(model)}")
        torch.save(model.state_dict(), "./ri_gpt.pth")

print("Training Done!")

print("Saving Trained Parameter Configuration...")
torch.save(model.state_dict(), "./ri_gpt.pth")
print("Trained Parameter Configuration Saved!")

print("Begining Inference...")
model.eval()
inputStr = 'Fang Yuan shouted with passion as he threw out a Gu Worm and yelled, "Liquor Worm Immortal Gu, Go!"'
out = model.generate(torch.tensor(tokenizer.encode(inputStr)).view(1, -1), 1000)
print(f"input: {tokenizer.decode(torch.tensor(tokenizer.encode(inputStr)).tolist()).decode("utf-8", "replace")}")
print(f"output: {tokenizer.decode(out[0, len(inputStr):].tolist()).decode("utf-8", "replace")}")
