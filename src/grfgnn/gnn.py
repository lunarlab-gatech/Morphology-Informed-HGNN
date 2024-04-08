from .datasets import CerberusStreetDataset
from .urdfParser import RobotURDF
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import MLP, GCN
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt


def test_model(model, evalLoader, device, A1_URDF, ground_truth_indices,
               mseLoss, epoch, indices):
    # Put model in evaluation mode
    model.eval()

    # Validate on entire eval_dataset
    with torch.no_grad():
        total_loss = 0
        pbar = tqdm(total=len(indices))
        pbar.set_description(f'Validating Epoch {epoch+1:02d}')
        for batch in evalLoader:

            # Run through the GCN
            out_raw = model(x=batch.x.to(device))  #,
            # edge_index=batch.edge_index.to(device)).squeeze()

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

            # Keep track of total loss
            total_loss += float(loss)
            pbar.update(batch.batch_size)
        pbar.close()

    # Put model back in training mode
    model.train()

    return total_loss


def train_model():
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
                                         batch_size=100,
                                         shuffle=True)
    testLoader: DataLoader = DataLoader(test_set, batch_size=100, shuffle=True)
    valLoader: DataLoader = DataLoader(val_set, batch_size=100, shuffle=True)

    # Set the current training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    # Create the model
    model = MLP(in_channels=dataset[0].x.flatten().shape[0],
                hidden_channels=256,
                num_layers=4,
                out_channels=4)
    #model = GCN(in_channels=dataset[0].x.shape[1],
    #            hidden_channels=256,
    #            num_layers=4,
    #            out_channels=1)
    model.to(device)

    # Use the Adam Optimizer with an Adaptive Learning Rate
    prev_lr = 0.03
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

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
    best_test_loss = None
    for epoch in range(0, epochs):
        pbar = tqdm(total=len(train_set.indices))
        pbar.set_description(f'Epoch {epoch+1:02d}')
        total_loss = 0
        for batch in trainLoader:
            # Reset the gradients
            optimizer.zero_grad()
            inputs = batch.x.flatten() # TAKE OUT FOR GCN
            print("Batch.x: ", batch.x)

            # Run through the GCN
            #out_raw = model(x=batch.x.to(device),
            #                edge_index=batch.edge_index.to(device)).squeeze()
            out_raw = model(x=inputs.to(device))
            print("out_raw: ", out_raw)

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
        test_loss = test_model(model, testLoader, device, A1_URDF,
                               ground_truth_indices, mseLoss, epoch,
                               test_set.indices) / len(test_set.indices)

        if epoch != 0:  # Can't do this before step due to a bug in Pytorch
            prev_lr = scheduler.get_last_lr()
        scheduler.step(test_loss)

        print("Loss: ", loss, "Test Loss: ", test_loss, "LR Used: ", prev_lr)

        # Save the model
        if best_test_loss is None or test_loss < best_test_loss:
            torch.save(
                model.state_dict(), "models/TestLoss:" + str(test_loss) +
                "_Epoch:" + str(epoch + 1) + "_trained_gnn.pth")
            best_test_loss = test_loss

    # output validation loss
    val_loss = test_model(model, valLoader, device, A1_URDF,
                          ground_truth_indices, mseLoss, epoch,
                          val_set.indices) / len(val_set.indices)
    print("Final Validation Loss: ", val_loss)

    # Save the model
    torch.save(model.state_dict(), "trained_gnn_100.pth")


def display_on_axes(axes, estimated, ground_truth, title):
    """
    Simple function that displays grounth truth and estimated
    information on a Matplotlib.pyplot Axes.
    """
    axes.plot(ground_truth, label="Ground Truth", linestyle='-.')
    axes.plot(estimated, label="Estimated")
    axes.legend()
    axes.set_title(title)


def display_viewable_graph():
    model = GCN(in_channels=2,
                hidden_channels=256,
                num_layers=3,
                out_channels=1)
    model.load_state_dict(
        torch.load("models/TestLoss:52.971655093561_Epoch:77_trained_gnn.pth"))
    model.eval()

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    model.to(device)

    # Calculate the indices in the output where we look for our
    # calculated forces
    ground_truth_indices = []
    name_to_index = A1_URDF.get_node_name_to_index_dict()
    for urdf_name in dataset.ground_truth_urdf_names:
        ground_truth_indices.append(name_to_index[urdf_name])
    print(ground_truth_indices)

    # Create a validation dataloader
    valLoader: DataLoader = DataLoader(val_set, batch_size=100, shuffle=False)

    # Setup four graphs
    fig, axes = plt.subplots(4, figsize=[20, 10])
    fig.suptitle('Foot Estimated Forces vs. Ground Truth')

    # Validate on entire eval_dataset
    with torch.no_grad():
        total_loss = 0
        for batch in valLoader:

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
            out_predicted = torch.stack(truth_tensors).swapaxes(0, 1).cpu()

            # Get the labels
            batch_y = torch.reshape(
                batch.y, (batch.batch_size, len(ground_truth_indices))).cpu()
            print(batch_y)
            print(batch_y.size())

            print(batch_y[:, 0].size())

            # Display the results
            titles = [
                "Front Left Foot Forces", "Front Right Foot Forces",
                "Rear Left Foot Forces", "Rear Right Foot Forces"
            ]
            for i in range(0, 4):
                display_on_axes(axes[i], out_predicted[:, i], batch_y[:, i],
                                titles[i])

            plt.savefig(f'test.png')
            break
