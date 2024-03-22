from datasets import CerberusStreetDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN
import torch


def main():
    # Load and shuffle the dataset
    dataset = CerberusStreetDataset(
        '/home/dlittleman/state-estimation-gnn/datasets/street.bag', 10)

    # Split the data into training, validation, and testing sets
    rand_gen = torch.Generator().manual_seed(10341885)
    train_set, test_set, val_set = torch.utils.data.random_split(
        dataset, [0.7, 0.2, 0.1], generator=rand_gen)

    # Create the dataloader and iterate through the batches
    trainloader: DataLoader = DataLoader(train_set, batch_size=5, shuffle=True)

    # Set the current training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    # Create the model
    model = GCN(in_channels=dataset[0].x.shape[1],
                hidden_channels=TODO,
                num_layers=TODO)
    model.to(device)


if __name__ == "__main__":
    main()
