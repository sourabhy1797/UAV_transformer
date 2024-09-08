import yaml
import torch
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset  # Dataset class from dataset.py
from model import TrajectoryTransformer  # Model class from model.py

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, -1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

    accuracy = correct / total * 100
    print(f'Accuracy: {accuracy:.2f}%')




if __name__ == '__main__':
    # Load configuration from YAML file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize dataset and dataloader
    dataset = TrajectoryDataset('data/nodes.csv', 'data/cleaned_data.csv')
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Initialize model
    model = TrajectoryTransformer(config, num_nodes=dataset.num_nodes, max_seq_len=dataset.max_seq_len)

    # Check for multi-GPU environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padded targets (-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))


    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, dataset.num_nodes), targets.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{config["training"]["epochs"]}], Loss: {loss.item():.4f}')


    # Evaluate the model
    # evaluate(model, dataloader)
