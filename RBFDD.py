'''
This is the implementation of the RBFDD network in PyTorch. The constructor needs
the number of Gaussians (i.e., h) and the dimensionality of the data that is being fed to the RBFDD network (i.e., pre_rbfdd_input_dim)
'''
class RBFDD(nn.Module):
    def __init__(self, h, pre_rbfdd_input_dim):
        super(RBFDD, self).__init__()
        # Number of hidden units (i.e., Gaussian Kernels)
        self.H = h
        # Dimensionality of the data that will be fed into the RBFDD network. (e.g., For 28x28 images, the dimensionality will be 784)
        self.pre_rbfdd_input_dim = pre_rbfdd_input_dim
        # Define the new parameters as class Parameter. The spread, Sd, and the centers, Mu, are the parameters of the Gaussians
        self.Sd = torch.nn.Parameter(torch.zeros((self.H,)))
        self.Mu = torch.nn.Parameter(torch.zeros((self.H, self.pre_rbfdd_input_dim)))
        # Define the linear layer, so you could have weights between the Gaussian layer and the output layer
        self.fc = nn.Linear(self.H, 1, bias=False)

    # Given the the current mini-batch of data, Gaussian function, computes the ouput of each of the Gaussians
    def Gaussian(self, x):
        p = torch.tensor([])
        a = x - self.Mu[:, None]
        b = torch.matmul(a, torch.transpose(a, dim0=1, dim1=2))
        numinator = torch.diagonal(b, dim1=1, dim2=2)
        spread_square = self.Sd ** 2
        denum = spread_square[:, None]
        power = -0.50 * torch.div(numinator, denum)
        power = self.clip_power(power) # Make sure you avoid rediculously large/low values for the power of the exponential.
        p = torch.exp(power)
        p = p.transpose(1, 0)
        return p


    # This function clipts ridiculously large/small values
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
