import os
import torch
from torch import nn
from torch.utils.data import DataLoader as dl
from torch.distributed.pipeline.sync import Pipe

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "2222"

inout_dim = 2
hidden_size = 16
datasize = 8000
batchsize = 4
logfreq = 100
device1 = "cuda:0"
device2 = "cuda:1"


def ground_truth(x):
    y = x.sum(dim=1, keepdim=True)
    y = y.expand_as(x)
    y = y * hidden_size
    return y

torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

fc1 = nn.Linear(inout_dim, hidden_size).to(device1)
fc2 = nn.Linear(hidden_size, inout_dim).to(device2)

fc1.weight.data.fill_(1.0)
fc2.weight.data.fill_(1.0)
fc1.bias.data.fill_(0.0)
fc2.bias.data.fill_(0.0)

model = nn.Sequential(fc1, fc2)
model = Pipe(model, chunks=8)
model.eval()

dataset = torch.randn((datasize, inout_dim))
dataloader = dl(dataset, batch_size=batchsize)

with torch.no_grad():
    for step, x in enumerate(dataloader):

        if step % logfreq == 0:
            print("Step " + str(step) + "/" + str(len(dataloader)))

        x = x.to(device1)
        y_pred = model(x).local_value()
        y_truth = ground_truth(x).to(device2)
        if step % logfreq == 0:
            print("Result Verified" if torch.
                  allclose(y_pred, y_truth, rtol=0.01) else "Error in Result")

print("2 GPUs, Pipeline Parallel Finished!")
