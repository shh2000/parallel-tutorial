import torch
from torch import nn
from torch.utils.data import DataLoader as dl

inout_dim = 2
hidden_size = 16
datasize = 256
batchsize = 4
logfreq = 10
device = "cuda:0"


class Matmul(nn.Module):

    def __init__(self):
        super(Matmul, self).__init__()
        self.fc1 = nn.Linear(inout_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, inout_dim)
        self.fc1.weight.data.fill_(1.0)
        self.fc2.weight.data.fill_(1.0)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.bias.data.fill_(0.0)

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
model.eval()

dataset = torch.randn((datasize, inout_dim))
dataloader = dl(dataset, batch_size=batchsize)

with torch.no_grad():
    for step, x in enumerate(dataloader):

        if step % logfreq == 0:
            print("Step " + str(step) + "/" + str(len(dataloader)))

        x = x.to(device)
        y_pred = model(x)
        y_truth = ground_truth(x)
        if step % logfreq == 0:
            print("Result Verified" if torch.
                  allclose(y_pred, y_truth, rtol=0.01) else "Error in Result")

print("Single GPU, No Parallel Finished!")
