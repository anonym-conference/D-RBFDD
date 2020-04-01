'''
D-RBFDD network is a mixture of LeNet-5 and RBFDD network. You need to import LeNEt5 and RBFDD classes.
'''
class D-RBFDD(nn.Module):
    # The constructor needs the number of Gaussians, h, and the dimensionality of the data that is going to be fed into the RBFDD network, pre_rbfdd_input_dim.
    def __init__(self, h, pre_rbfdd_input_dim):
        super(D-RBFDD, self).__init__()
        self.H = h
        self.pre_rbfdd_input_dim = pre_rbfdd_input_dim
        # Create the LeNet-5 network
        self.lenet = LeNet5()
        # create the RBFDD network
        self.rbfdd = RBFDD(h, pre_rbfdd_input_dim)
    def forward(self, x):
        x = self.lenet(x)
        # APPLY Lecun's non-linearity here
        x = 1.7159 * torch.tanh(float(2 / 3) * x)
        # pass the LeNEt's output through the RBFDD network
        x = self.rbfdd(x)
        return x
    # This function will only transfer the data through the deep portion of the network (not the RBFDD). Used for
    # Initializing the Gaussians using the whole training data transform into a latent representation. K-means will run
    # On the output of this function to initialize the Gaussians centers and spreads.
    def partial_forward(self, x):
        x = self.lenet(x)
        # APPLY Lecun's non-linearity here
        x = 1.7159 * torch.tanh(float(2 / 3) * x)
        return x
