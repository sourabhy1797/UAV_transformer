import yaml
import torch
from torch.utils.data import DataLoader, random_split
from dataset import TrajectoryDataset  # Dataset class from dataset.py
from model import TrajectoryTransformer  # Model class from model.py

# Evaluation function
def evaluate(model, val_loader):
# Load the best model for evaluation
    print("Loading the best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path))

    # Evaluation loop (on validation set or test set)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:  # Using validation set for evaluation
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, -1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

    evaluation_accuracy = correct / total * 100
    print(f'Final Evaluation Accuracy: {evaluation_accuracy:.2f}%')




if __name__ == '__main__':
    # Load configuration from YAML file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize dataset
    dataset = TrajectoryDataset('data/nodes.csv', 'data/cleaned_data.csv')

    # Split dataset into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Initialize model with dynamic num_nodes and max_seq_len directly from dataset
    model = TrajectoryTransformer(config, num_nodes=dataset.num_nodes, max_seq_len=dataset.max_seq_len)

    # Check for multi-GPU environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padded targets (-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))

    # Variables to keep track of the best validation accuracy and save model
    best_val_accuracy = 0.0
    best_model_path = "best_model.pth"

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, dataset.num_nodes), targets.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{config["training"]["epochs"]}], Train Loss: {train_loss/len(train_loader):.4f}')

        # Validation loop
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                val_loss += criterion(outputs.view(-1, dataset.num_nodes), targets.view(-1)).item()

                _, predicted = torch.max(outputs, -1)
                correct += (predicted == targets).sum().item()
                total += targets.numel()

        val_accuracy = correct / total * 100
        print(f'Epoch [{epoch + 1}/{config["training"]["epochs"]}], Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%')

        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {val_accuracy:.2f}%")

    # Load the best model for evaluation
    print("Loading the best model for evaluation...")


    # Evaluate the model
    evaluate(model, val_loader)
