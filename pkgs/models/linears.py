from torch.nn import Module, Linear


class LinearModel(Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = Linear(3, 1) # day,month,year to meld

    def forward(self, x):
        x = self.fc(x)
        return x