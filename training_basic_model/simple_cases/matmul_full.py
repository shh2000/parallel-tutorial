import torch
from torch import nn
from torch.utils.data import DataLoader as dl
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

inout_dim = 2
hidden_size = 16
datasize = 8000
batchsize = 4
logfreq = 500
device = "cuda:0"

num_epochs = 20
base_lr = 0.01
lr_gamma = 0.1
lr_step = 5


class Matmul(nn.Module):

    def __init__(self):
        super(Matmul, self).__init__()
        self.fc1 = nn.Linear(inout_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, inout_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def ground_truth(x):
    y = x.sum(dim=1, keepdim=True)
    y = y.expand_as(x)
    y = y * hidden_size
    return y


model = Matmul().to(device)

dataset = torch.randn((datasize, inout_dim))
dataloader = dl(dataset, batch_size=batchsize)

criterion = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=base_lr)
scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

model.train()
for epoch in range(num_epochs):
    for step, x in enumerate(dataloader):
        optimizer.zero_grad()

        x = x.to(device)
        y_pred = model(x)
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

        x = x.to(device)
        y_pred = model(x)
        y_truth = ground_truth(x)
        if step % logfreq == 0:
            print("Result Verified @ 99% precision" if torch.
                  allclose(y_pred, y_truth, rtol=0.01) else "Error in Result")

print("Single GPU, No Parallel Finished!")
