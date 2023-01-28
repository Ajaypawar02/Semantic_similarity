import torch.nn as nn


def loss_fn(out, tar):
#     print(out)
#     print(tar)
    loss = nn.BCEWithLogitsLoss()(out, tar.view(-1, 1))
    return loss

# print(loss_fn(out.flatten(), tar))