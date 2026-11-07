import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")
# DATA AUGMENTATION + NORMALIZATION


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

#  LOAD CIFAR-10


trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=train_transform
)

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=test_transform
)


# 3. DATA LOADERS

trainloader = DataLoader(
    trainset,
    batch_size=128,
    shuffle=True
)

testloader = DataLoader(
    testset,
    batch_size=128,
    shuffle=False
)

# 4. DEFINE 6-LAYER VGG-STYLE CNN

class VGG6CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            64, 64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)


        # Block 2
        self.conv3 = nn.Conv2d(
            64, 128,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
            128, 128,
            kernel_size=3,
            padding=1
        )

        self.bn4 = nn.BatchNorm2d(128)


        # Block 3
        self.conv5 = nn.Conv2d(
            128, 256,
            kernel_size=3,
            padding=1
        )

        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(
            256, 256,
            kernel_size=3,
            padding=1
        )

        self.bn6 = nn.BatchNorm2d(256)


        # Pooling layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))


        # Final classifier
        self.fc = nn.Linear(
            256,
            10
        )


    def forward(self, x):

        # Block 1
        # Input: 3 x 32 x 32


        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.pool(x)

        # Shape:
        # 64 x 16 x 16


        # Block 2


        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.pool(x)

        # Shape:
        # 128 x 8 x 8


        # Block 3


        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)

        x = self.pool(x)

        # Shape:
        # 256 x 4 x 4


        # Global average pooling


        x = self.gap(x)

        # Shape:
        # 256 x 1 x 1


        # Flatten
        x = torch.flatten(x, 1)

        # Shape:
        # 256


        # Final output
        x = self.fc(x)

        # Shape:
        # 10

        return x


# 5. DEVICE


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

# 6. CREATE MODEL

model = VGG6CNN().to(device)

print(model)


# 7. LOSS FUNCTION


criterion = nn.CrossEntropyLoss()

# 8. OPTIMIZER


optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)


# 9. LEARNING RATE SCHEDULER


scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[80, 120, 160],
    gamma=0.1
)

# 10. TRAINING


num_epochs = 100 

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in trainloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    scheduler.step()


    train_accuracy = (
        100 * correct / total
    )

    average_loss = (
        running_loss
        / len(trainloader)
    )


    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Loss: {average_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )


# 11. TEST ACCURACY

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in testloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


test_accuracy = (
    100 * correct / total
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy:.2f}%"
)


# 12. SAVE TRAINED MODEL


torch.save(
    model.state_dict(),
    "vgg6_cifar10.pth"
)

print(
    "\nModel saved as "
    "vgg6_cifar10.pth"
)


# 13. INSPECT FIRST CONVOLUTION WEIGHTS


print(
    "\nConv1 weight shape:",
    model.conv1.weight.shape
)

print(
    "\nFirst Conv1 filter:"
)

print(
    model.conv1.weight[0]
)