class LeNet5(nn.Module):
    def __init__(self):

        super(LeNet5, self).__init__()
        
        self.cnn_model = nn.Sequential(
        nn.Conv2d(in_channel, 6, kernel_size=5)
        nn.Tanh(),
        nn.AvgPool2d(2, stride=2)
        nn.Conv2d(6, 16, kernel_size=5)
        nn.Tanh(),
        nn.AvgPool2d(2, stride=2))
        self.fc_model = nn.Sequential(
            nn.Linear(flattened, 120),
            nn.Tanh(),
            nn.Linear(120, 84)) # Note that we have eliminated the last tanh(). As the D-RBFDD network, apply a tanh() on the output of LeNet anyways


    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)

        return x
