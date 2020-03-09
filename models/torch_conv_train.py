import itertools

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Quant(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, quant_param):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        ctx.save_for_backward(Variable(torch.ones(1, 1), requires_grad=False))
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input < 0] = 0
        return grad_input


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

# def convQuant(convLayer: nn.Module):
#     convLayer()
#     y_pred = relu(x.mm(w1)).mm(w2)
#
#     # Compute and print loss
#     loss = (y_pred - y).pow(2).sum()
#     if t % 100 == 99:
#         print(t, loss.item())
#
#     # Use autograd to compute the backward pass.
#     loss.backward()
#
#     # Update weights using gradient descent
#     with torch.no_grad():
#         w1 -= learning_rate * w1.grad
#         w2 -= learning_rate * w2.grad
#
#         # Manually zero the gradients after updating weights
#         w1.grad.zero_()
#         w2.grad.zero_()
QuantizationCrunch = {}


def forward_hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
    # print("Forward Module : {}. hash {}".format(module, module.__hash__()))
    # for layer in module.modules():
    if isinstance(module, nn.Conv2d):
        # print("Layer dict {}".format(module.state_dict().keys()))
        # str = "quant_{}_input".format(list(module.named_modules())
        # module.register_parameter("orig_input", input[0])
        # module.register_buffer("orig_output", output[0])
        QuantizationCrunch[str(module.__hash__())] = {"input": input[0], "output": output[0]}  # module.


def backward_hook(module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
    # print("Backward Module : {}, grad_inp: {}, grad_out: {}".format(module, len(grad_input), len(grad_output)))
    # Forward pass: compute predicted y using operations; we compute
    # ReLU using our custom autograd operation.
    if isinstance(module, nn.Conv2d):
        inp = QuantizationCrunch[str(module.__hash__())]["input"]
        # module.register_buffer("orig_output", output)
        # torch.utils.hooks.RemovableHandle(clone).remove()
        # print("w: {}, b: {}".format(module.weight, module.bias))
        quant_out, quant_weights = quantize(module, module.weight, module.bias, inp)
        # module.weight = torch.nn.Parameter(quant_weights)


def quantize(mod: torch.nn.Module, weights: torch.Tensor, bias: torch.Tensor, inp: torch.Tensor) -> {torch.Tensor,
                                                                                                     torch.Tensor}:
    orig_out = mod.forward(inp)
    weights_shape = weights.shape
    l_w = weights.flatten().tolist()
    for i in range(len(l_w)):
        if l_w[i] * 10_000 % 1 > 0:
            # print("vl {} : {}".format(l_w[i], int(l_w[i] * 1_000) / 1_000.))
            l_w[i] = int(l_w[i] * 10_000) / 10_000.
        # if l_w[i] != 0:
        #     l_w[i] = 0.
    q_w = torch.as_tensor(l_w).requires_grad_(False).reshape(weights_shape)
    # print("q_w : {}".format(q_w))
    mod.weight = torch.nn.Parameter(q_w, True)
    quant_out = mod.forward(inp)
    plt.plot(range(len(orig_out.flatten().tolist())), (quant_out - orig_out).flatten().tolist(), ",")
    # print("Diff {}".format(quant_out - orig_out))
    QuantizationCrunch.pop(str(mod.__hash__()))
    return [quant_out, q_w]


criterion = nn.CrossEntropyLoss()
# print(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.001) # , momentum=0.9)

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in itertools.islice(enumerate(trainloader, 0), 25):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # state = net.state_dict()
        # print("state dict: {}".format(state))
        net.conv1.register_forward_hook(hook=forward_hook)
        net.conv1.register_backward_hook(hook=backward_hook)
        # net.conv2.register_forward_hook(hook=forward_hook)
        # net.conv2.register_backward_hook(hook=backward_hook)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # for layer in net.modules():
        #     # print("\tModules {} ".format(layer))
        #     if isinstance(layer, nn.Conv2d):
        #         print("Layer dict {}".format(layer.state_dict().keys()))
        #         w1 = layer.state_dict().get("weights")
        #         learning_rate = 0.01
        #         with torch.no_grad():
        #             w1 -= learning_rate * w1.grad
        #
        #             # Manually zero the gradients after updating weights
        #             w1.grad.zero_()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0

plt.show()
print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

l_w = net.conv1.weight.flatten().tolist()
for i in range(len(l_w)):
    if l_w[i] * 10_000 % 1 > 0:
        # print("vl {} : {}".format(l_w[i], int(l_w[i] * 1_000) / 1_000.))
        print("not worked")

print("w1 : {}".format(
    net.conv1.weight.tolist()
))
print("w2 : {}".format(
    net.conv2.weight.tolist()
))
print("Accuracy of network on the 10_000 test images: %d %%" % (100 * correct / total))


