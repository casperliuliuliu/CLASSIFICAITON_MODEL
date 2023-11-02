import torch.nn as nn

# Setup
modelA = nn.Linear(1, 1)
modelB = nn.Linear(1, 1)
print(modelA.state_dict())
print(modelB.state_dict())

sdA = modelA.state_dict()
sdB = modelB.state_dict()

# Average all parameters
for key in sdA:
    print(key)
    sdB[key] = (sdB[key] + sdA[key]) / 2.

# Recreate model and load averaged state_dict (or use modelA/B)
model = nn.Linear(1, 1)
model.load_state_dict(sdB)
print(model.state_dict())
