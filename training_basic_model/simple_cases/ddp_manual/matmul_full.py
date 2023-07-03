import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler as distSampler

inout_dim = 2
hidden_size = 16
datasize = 8000
batchsize = 4
logfreq = 50

num_epochs = 30
base_lr = 0.01
lr_gamma = 0.1
lr_step = 10


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


class MyDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_and_save_dataset():
    dataset = MyDataset(torch.randn((datasize, inout_dim)))
    torch.save(dataset, "dataset.pt")


def inference(rank):
    dataset = torch.load("dataset.pt")
    dataloader = DataLoader(dataset,
                            batch_size=batchsize,
                            num_workers=8,
                            sampler=distSampler(dataset))

    model = Matmul().to(rank)
    
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=base_lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    model.train()
    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)
        for step, x in enumerate(dataloader):
            optimizer.zero_grad()

            x = x.to(rank)
            y_pred = model(x)
            y_truth = ground_truth(x)

            loss = criterion(y_pred, y_truth)
            loss.backward()

            for param in model.parameters():
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= 8
            optimizer.step()

            if step % logfreq == 0 and rank == 0:
                print("Epoch [{}/{}], Step [{}/{}]: loss = {}".format(
                    epoch + 1, num_epochs, step, len(dataloader), loss))
        scheduler.step()

    model.eval()
    with torch.no_grad():
        for step, x in enumerate(dataloader):
            if step % logfreq == 0 and rank == 0:
                print("Step " + str(step) + "/" + str(len(dataloader)))

            x = x.to(rank)
            y_pred = model(x)
            y_truth = ground_truth(x)

        dist.barrier()

        print("Rank " + str(rank) + ": " + "Result Verified" if torch.
              allclose(y_pred, y_truth, rtol=0.01) else "Error in Result")

    dist.barrier()


def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        create_and_save_dataset()

    dist.barrier()

    inference(rank)

    if rank == 0:
        print("8 GPU, Data Parallel Finished!")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
