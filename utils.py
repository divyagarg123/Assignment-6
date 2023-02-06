import torch
from torchsummary import summary


def check_for_cuda():
    SEED = 1
    cuda = torch.cuda.is_available()
    print("CUDA Available", cuda)
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)
    return cuda


def print_model_summary(model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    summary(model, input_size=(3, 32, 32))
