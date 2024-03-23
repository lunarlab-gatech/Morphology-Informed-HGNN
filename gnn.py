from datasets import CerberusStreetDataset
from urdfParser import RobotURDF
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def main():
    # Load the A1 urdf
    A1_URDF = RobotURDF('urdf_files/A1/a1.urdf', 'package://a1_description/',
                        'unitree_ros/robots/a1_description', True)

    # Load and shuffle the dataset
    dataset = CerberusStreetDataset(
        '/home/dlittleman/state-estimation-gnn/datasets/cerberus_street',
        A1_URDF, None)

    # Split the data into training, validation, and testing sets
    rand_gen = torch.Generator().manual_seed(10341885)
    train_set, test_set, val_set = torch.utils.data.random_split(
        dataset, [0.7, 0.2, 0.1], generator=rand_gen)

    # Create the dataloader and iterate through the batches
    trainLoader: DataLoader = DataLoader(train_set,
                                         batch_size=32,
                                         shuffle=False)

    # Set the current training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    # Create the model
    model = GCN(in_channels=dataset[0].x.shape[1],
                hidden_channels=256,
                num_layers=3,
                out_channels=1)
    model.to(device)

    # Use the Adam Optimizer with an Adaptive Learning Rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=7)

    # Use MSELoss
    mseLoss = torch.nn.MSELoss()

    # Define epochs to train
    epochs = 100

    # Calculate the indices in the output where we look for our
    # calculated forces
    ground_truth_indices = []
    name_to_index = A1_URDF.get_node_name_to_index_dict()
    for urdf_name in dataset.ground_truth_urdf_names:
        ground_truth_indices.append(name_to_index[urdf_name])
    print(ground_truth_indices)

    # Run the training loop
    model.train()
    for epoch in range(1, epochs):
        pbar = tqdm(total=len(train_set.indices))
        pbar.set_description(f'Epoch {epoch:02d}')
        total_loss = total_correct = 0
        for batch in trainLoader:
            # Reset the gradients
            optimizer.zero_grad()

            # Run through the GCN
            out_raw = model(x=batch.x.to(device),
                            edge_index=batch.edge_index.to(device)).squeeze()

            # Reshape so that we have a tensor of (num_nodes, batch_size)
            out_nodes_by_batch = torch.reshape(
                out_raw, (batch.batch_size, A1_URDF.get_num_nodes()))

            # Get the outputs from the foot nodes
            truth_tensors = []
            for index in ground_truth_indices:
                truth_tensors.append(out_nodes_by_batch[:, index])
            out_predicted = torch.stack(truth_tensors).swapaxes(0, 1)

            # Get the labels
            batch_y = torch.reshape(
                batch.y,
                (batch.batch_size, len(ground_truth_indices))).to(device)

            # Calculate the loss
            loss = mseLoss(out_predicted, batch_y)

            # Update the model
            loss.backward()
            optimizer.step()

            # Keep track of total loss
            total_loss += float(loss)
            pbar.update(batch.batch_size)

        pbar.close()
        loss = total_loss / len(train_set.indices)
        #approx_acc = total_correct / train_idx.size(0)
        #train_acc, val_acc, test_acc, var = test(model, device)

        print("Loss: ", loss)
        #print(
        #    f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Var: {var:.4f}'
        #)


if __name__ == "__main__":
    main()
