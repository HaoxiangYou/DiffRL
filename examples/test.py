import torch
import torch.nn as nn
import os
import random
import numpy as np 

sequence_size = 32
env_size = 64
input_dim = 39
hidden_dim = 64
output_dim = 6
device = "cuda:0"
seed = 0

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


batch_input = torch.randn((sequence_size, env_size, input_dim), dtype=torch.float64, device=device)

model = nn.Linear(in_features=input_dim, out_features=output_dim, device=device).double()
batch_output = model(batch_input)

print("big batch together:", batch_output[0,0])
print("smaller batch:", model(batch_input[0])[0])
