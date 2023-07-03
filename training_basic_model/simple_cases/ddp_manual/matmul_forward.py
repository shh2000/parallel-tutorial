import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset

inout_dim = 2
hidden_size = 16
datasize = 8000
batchsize = 4
logfreq = 50


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
    rank_length = int(datasize / 8)
    rank_dataset = Subset(dataset,
                          range(rank_length * rank, rank_length * (rank + 1)))
    dataloader = DataLoader(rank_dataset, batch_size=batchsize, num_workers=8)
    model = Matmul().to(rank)
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
