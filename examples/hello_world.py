import torch
import torch.nn.functional as F
from torch.nn import Conv2d, ReLU, Sequential
from octoml_profile import (accelerate,
                            remote_profile,
                            RemoteInferenceSession)

model = Sequential(Conv2d(16, 16, 3), ReLU())

@accelerate
def predict(x: torch.Tensor):
    print("enter predict function")
    y = model(x)
    z = F.softmax(y)
    print("exit predict function")
    return z

# Alternatively you can also directly use `accelerate`
# on a model, e.g. `predict = accelerate(model)`

session = RemoteInferenceSession()
with remote_profile(session):
    x = torch.randn(1, 16, 20, 20)
    for i in range(10):
        predict(x)
