from lagom.utils import set_global_seeds
set_global_seeds(seed=0)

import numpy as np

import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lagom.utils import numpify
from lagom.networks import Module
from lagom.networks import make_fc
from lagom.networks import ortho_init
from lagom.networks import MDNHead


class Dataset(data.Dataset):
    r"""Generate a set of data point of an inverted sinusoidal function. 
    i.e. y(x) = 7sin(0.75x) + 0.5x + eps, eps~N(0, 1)
    
    Then we ask the neural networks to predict x given y, in __getitem__(). 
    In this case, the classic NN suffers due to only one output given input. 
    To address it, one can use Mixture Density Networks. 
    """
    def __init__(self, n):
        self.n = n
        self.x, self.y = self._generate_data(self.n)
    
    def _generate_data(self, n):
        eps = np.random.randn(n)
        x = np.random.uniform(low=-10.5, high=10.5, size=n)
        y = 7*np.sin(0.75*x) + 0.5*x + eps
        
        return np.float32(x), np.float32(y)  # Enforce the dtype to be float32, i.e. FloatTensor in PyTorch
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        # Retrieve the x, y value
        x = self.x[index]
        y = self.y[index]
        # Keep array shape due to scalar value
        x = np.array([x], dtype=np.float32)
        y = np.array([y], dtype=np.float32)

        return y, x


class MDN(Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_density, **kwargs):
        super().__init__(**kwargs)
        
        self.hidden_sizes = hidden_sizes
        self.feature_layers = make_fc(input_size, hidden_sizes)
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in hidden_sizes])
        self.mdn_head = MDNHead(hidden_sizes[-1], output_size, num_density)

    def forward(self, x):
        for layer, layer_norm in zip(self.feature_layers, self.layer_norms):
            x = layer_norm(F.relu(layer(x)))
        return self.mdn_head(x)


device = torch.device('cpu')
model = MDN(input_size=1, hidden_sizes=[15, 15], output_size=1, num_density=20).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)

#model


N = 5000
epochs = 1000
batch_size = 128

dataset = Dataset(n=N)
dataloader = data.DataLoader(dataset, batch_size=batch_size)
model.train()
for epoch in range(epochs):
    losses = []
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        logit_pi, mean, std = model(x)
        loss = model.mdn_head.loss(logit_pi, mean, std, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
    if epoch == 0 or (epoch+1)%100 == 0:
        #IPython.display.clear_output(wait=True)
        print(f'Epoch: {epoch+1}\t Loss: {np.mean(losses)}')

test_data = torch.linspace(-15, 15, steps=1000).unsqueeze(1)

model.eval()
logit_pi, mean, std = model(test_data.to(device))
samples = model.mdn_head.sample(logit_pi, mean, std, 2.0)

print(f"\n\n\nScore: {(torch.mean(samples)).detach().cpu().numpy()}\n\n\n")
