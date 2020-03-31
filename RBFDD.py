class RBFDD(nn.Module):
    def __init__(self, h, pre_rbfdd_input_dim):
        # Number of hidden units (i.e., Gaussian Kernels)
        super(RBFDD, self).__init__()
        # Number of hidden units (i.e., Gaussian Kernels)
        self.H = h
        self.pre_rbfdd_input_dim = pre_rbfdd_input_dim
        # Define the new parameters as class Parameter
        self.Sd = torch.nn.Parameter(torch.zeros((self.H,)))
        self.Mu = torch.nn.Parameter(torch.zeros((self.H, self.pre_rbfdd_input_dim)))
        # Define the linear layer, so you could have weights beteen the Gaussian layer and the output layer
        self.fc = nn.Linear(self.H, 1, bias=False)


    def Gaussian(self, x):
        div = False
        p = torch.tensor([])
        try:
            a = x - self.Mu[:, None]
            b = torch.matmul(a, torch.transpose(a, dim0=1, dim1=2))
            numinator = torch.diagonal(b, dim1=1, dim2=2)
            spread_square = self.Sd ** 2
            denum = spread_square[:, None]
            power = -0.50 * torch.div(numinator, denum)
            power = self.clip_power(power)
            p = torch.exp(power)
            p = p.transpose(1, 0)
        except RuntimeError:
            div = True
        return p, div



    def clip_power(self, power):
        minimum = torch.tensor(-100.).to(device)
        maximum = torch.tensor(40.).to(device)
        power = torch.where(power < minimum, minimum, power)
        power = torch.where(power > maximum, maximum, power)

        return power

    def forward(self, x):
        y = torch.tensor([])
        [p, div] = self.Gaussian(x)
        if div is True:
            return y, div

        z = self.fc(p)
        y = 1.7159 * torch.tanh(float(2 / 3) * z)
        return y, div
