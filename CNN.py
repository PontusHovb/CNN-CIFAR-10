import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time

BATCH_SIZE = 8
POOL_SIZE = 2
CONV1 = 32
CONV2 = 32


start_time = time.time()

# Data Preparation
# Define data transformations, load and preprocess the dataset (e.g., CIFAR-10)
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load and preprocess the dataset
torchvision.datasets.CIFAR10.url="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)

# Define a simple CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=CONV1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=CONV1, out_channels=CONV2, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)

        # Calculate the dimensionality after pooling
        self.pool_output_size = CONV2 * 224 // (POOL_SIZE**2) * 224 // (POOL_SIZE**2)

        # Define fully connected layers
        self.fc1 = nn.Linear(self.pool_output_size, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, self.pool_output_size)
        x = self.fc1(x)
        return x

net = Net()


# Use a previously trained model
net = Net()
net.load_state_dict(torch.load('cnn_model_ver2.pth'))
"""
# Use a pretrained model
net = models.alexnet(pretrained=False)
model_path = filepath.alexnet_filepath()
net.load_state_dict(torch.load(model_path))
num_ftrs = net.classifier[6].in_features
net.classifier[6] = nn.Linear(num_ftrs, 10)  # Assuming your task has 10 classes
"""

def training():
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    start_time = time.time()

    # Training
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {round(running_loss / 100,3)}, time: {round(time.time()-start_time,1)}s, progress: {100*i*BATCH_SIZE/len(trainset)}%')
                running_loss = 0.0
                start_time = time.time()

    torch.save(net.state_dict(), 'cnn_model.pth')
    print('Finished Training')

def evaluation():
    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

#training()
evaluation()