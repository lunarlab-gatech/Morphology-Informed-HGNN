from datasets import CerberusStreetDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN


def main():
    # Load and shuffle the dataset
    dataset = CerberusStreetDataset(
        '/home/dlittleman/state-estimation-gnn/datasets/street.bag', 10000)

    # Create the dataloader and iterate through the batches
    loader: DataLoader = DataLoader(dataset,
                                    batch_size=5,
                                    shuffle=True,
                                    num_workers=8)
    for batch in loader:
        print(batch)


if __name__ == "__main__":
    main()
