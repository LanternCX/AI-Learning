import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# 定义数据预处理流程
# 将图像转换为Tensor，并归一化到[-1,1]区间
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载并加载训练集数据
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 使用DataLoader封装训练集，设置批量大小为64，数据打乱顺序
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 下载并加载测试集数据
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 使用DataLoader封装测试集，批量大小64，顺序读取
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # 第一层卷积层，输入通道数1，输出通道数6，卷积核大小5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)

        # 平均池化层，池化窗口2x2，步长2
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 第二层卷积层，输入通道数6，输出通道数16，卷积核大小5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # 第一个全连接层，输入维度16*4*4，输出维度120
        self.fc1 = nn.Linear(16 * 4 * 4, 120)

        # 第二个全连接层，输入120，输出84
        self.fc2 = nn.Linear(120, 84)

        # 第三个全连接层，输入84，输出10（对应10个数字类别）
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 第一层卷积，后接ReLU激活，再池化
        x = self.pool(torch.relu(self.conv1(x)))

        # 第二层卷积，后接ReLU激活，再池化
        x = self.pool(torch.relu(self.conv2(x)))

        # 将多维张量展平为二维，送入全连接层
        x = x.view(-1, 16 * 4 * 4)

        # 第一个全连接层，后接ReLU激活
        x = torch.relu(self.fc1(x))

        # 第二个全连接层，后接ReLU激活
        x = torch.relu(self.fc2(x))

        # 最后一个全连接层，输出类别得分
        x = self.fc3(x)

        return x


def train(net, device, trainloader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):

        # 将模型设置为训练模式
        net.train()

        running_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader):

            # 将输入和标签数据发送到指定设备
            inputs, labels = inputs.to(device), labels.to(device)

            # 清空优化器中的梯度
            optimizer.zero_grad()

            # 前向传播计算输出
            outputs = net(inputs)

            # 计算损失值
            loss = criterion(outputs, labels)

            # 反向传播计算梯度
            loss.backward()

            # 更新模型参数
            optimizer.step()

            running_loss += loss.item()

            # 每100个批次打印一次平均损失
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0


def test(net, device, testloader):

    # 将模型设置为评估模式
    net.eval()

    correct = 0
    total = 0

    # 测试时关闭梯度计算，节省内存和计算
    with torch.no_grad():
        for inputs, labels in testloader:

            # 将输入和标签数据发送到指定设备
            inputs, labels = inputs.to(device), labels.to(device)

            # 计算模型输出
            outputs = net(inputs)

            # 取输出中最大值对应的索引作为预测类别
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            # 累加预测正确的样本数
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy


def main():

    # 判断是否有GPU可用，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # 实例化网络，并发送到指定设备
    net = LeNet().to(device)

    # 定义损失函数为交叉熵
    criterion = nn.CrossEntropyLoss()

    # 定义优化器为带动量的SGD
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    print("Start training...")

    # 训练模型
    train(net, device, trainloader, criterion, optimizer, epochs=5)

    print("Start testing...")

    # 测试模型
    test(net, device, testloader)


if __name__ == "__main__":
    main()
