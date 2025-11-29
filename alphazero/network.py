import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_SPACE = 12
## Convolutional Layer
class ConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
    def forward(self, x):
        x = self.conv(x)
        return F.relu(self.bn(x))
## Residual Blocks (with BatchNorm and ReLU)
class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)
    
class OutBlock(nn.Module):
    def __init__(self, action_space=ACTION_SPACE):
        super().__init__()
        # Value Head
        self.conv1 = nn.Conv2d(64, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8 + 32, 64)
        self.fc2 = nn.Linear(64, 1)

        #Policy Head
        self.conv2 = nn.Conv2d(64, 32, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc3 = nn.Linear(32*8*8, ACTION_SPACE)
    def forward(self, x, scalar_embed):
        v = F.relu(self.bn(self.conv1(x)))
        v = v.view(v.size(0), -1)
        v = torch.cat([v, scalar_embed], dim=1)   # inject scalar info
        v = F.relu(self.fc1(v))
        value = torch.tanh(self.fc2(v))

        p = F.relu(self.bn2(self.conv2(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.fc3(p)

        return value, policy_logits

class ScalarFeaturesHead(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Network(nn.Module):
  def __init__(self, num_res_blocks=8):
      super().__init__()
      self.conv = ConvLayer()

      self.res_blocks = nn.ModuleList([ResidualBlock() for _ in range(num_res_blocks)])

      self.scalar_head = ScalarFeaturesHead()

      self.out_block = OutBlock()

  def forward(self, image, scalar_features):
    x = self.conv(image)
    for block in self.res_blocks:
        x = block(x)
    scalar_out = self.scalar_head(scalar_features)
    value, policy = self.out_block(x, scalar_out)

    return value, policy
  def inference(self, image, scalar_features):
      self.eval()
      with torch.no_grad():
            value, logits = self.forward(image, scalar_features)
            policy = F.softmax(logits, dim=1)
            return value, policy
  def get_weights(self):
    # Returns the weights of this network.
    return self.state_dict()
def make_uniform_network():
    return Network()
