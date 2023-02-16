import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from octoml_profile import (accelerate,
                            remote_profile,
                            RemoteInferenceSession)

model = Sequential(Linear(100, 200), ReLU(), Linear(200, 10))

@accelerate
def predict(x: torch.Tensor):
    y = model(x)
    z = F.softmax(y, dim=-1)
    return z

# Alternatively you can also directly use `accelerate`
# on a model, e.g. `predict = accelerate(model)` which will leave the
# softmax out of remote execution

session = RemoteInferenceSession()
with remote_profile(session):
    for i in range(10):
        x = torch.randn(1, 100)
        predict(x)
