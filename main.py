# Importing External Libs
import torch
from torch.optim import AdamW, lr_scheduler

# Custom Libs
from libs.globals import HyprParams
from libs.model import Model
from libs.splitDataset import BPE
from libs.utils import loadBatch, estimateLoss, count_model_parameters

# Order in the bleak world ruled by Chaos
gen = torch.manual_seed(12345)

model = Model().to(HyprParams.device)
print(f"Total Trainable Params: {count_model_parameters(Model())}")

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
inputStr = '	He laughed ruthlessly as he said: "Star Constellation Immortal Venerable, I am going to kill all of Heavenly Court\'s'
out = model.generate(torch.tensor(BPE.encode(inputStr)).view(1, -1).to(HyprParams.device), 1000)
print(f"input: {BPE.decode(torch.tensor(BPE.encode(inputStr)).tolist())}")
print(f"output: {BPE.decode(out[0, len(inputStr):].tolist())}")
