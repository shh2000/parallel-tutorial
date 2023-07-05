import os
import torch
from torch import nn
from torch.utils.data import DataLoader as dl
from torch.distributed.pipeline.sync import Pipe
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "2222"

inout_dim = 2
hidden_size = 16
datasize = 8000
batchsize = 4
logfreq = 500
device1 = "cuda:0"
device2 = "cuda:1"

num_epochs = 1
base_lr = 0.01
lr_gamma = 0.1
lr_step = 5


def ground_truth(x):
    y = x.sum(dim=1, keepdim=True)
    y = y.expand_as(x)
    y = y * hidden_size
    return y


torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

fc1 = nn.Linear(inout_dim, hidden_size).to(device1)
fc2 = nn.Linear(hidden_size, inout_dim).to(device2)

model = nn.Sequential(fc1, fc2)
model = Pipe(model, chunks=8)

dataset = torch.randn((datasize, inout_dim))
dataloader = dl(dataset, batch_size=batchsize)

criterion = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=base_lr)
scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

model.train()
for epoch in range(num_epochs):
    for step, x in enumerate(dataloader):
        optimizer.zero_grad()

        x = x.to(device1)
        y_pred = model(x).local_value().to(device1)
        y_truth = ground_truth(x)

        loss = criterion(y_pred, y_truth)
        loss.backward()
        optimizer.step()

        if step % logfreq == 0:
            print("Epoch [{}/{}], Step [{}/{}]: loss = {}".format(
                epoch + 1, num_epochs, step, len(dataloader), loss))
    scheduler.step()

model.eval()
with torch.no_grad():
    for step, x in enumerate(dataloader):

        if step % logfreq == 0:
            print("Step " + str(step) + "/" + str(len(dataloader)))

        x = x.to(device1)
        y_pred = model(x).local_value().to(device1)
        y_truth = ground_truth(x)
        if step % logfreq == 0:
            print("Result Verified @ 99% precision" if torch.
                  allclose(y_pred, y_truth, rtol=0.01) else "Error in Result")

print("2 GPUs, Pipeline Parallel Finished!")
