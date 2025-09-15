from model_action.arch import TCNNetwork
import torch

if __name__ == '__main__':
    model = TCNNetwork()

    input = torch.rand((8,16,48))
    x, y = model(input)
    print(x.shape)
    print(len(y),y[0].shape)
