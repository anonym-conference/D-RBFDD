class D-RBFDD(nn.Module):
    def __init__(self, h, pre_rbfdd_input_dim, fine_tune=False, fine_tune_layers = [], DeepNetOption = 'ResNet'):
        super(D-RBFDD, self).__init__()
        # Load the pre-trained model
        self.H = h
        self.pre_rbfdd_input_dim = pre_rbfdd_input_dim
        self.DeepNetOption = DeepNetOption
        if self.DeepNetOption == 'ResNet':
            self.resnet = models.resnet18(pretrained=True)

            # Convert it to single channel
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


            self.fine_tune = fine_tune
            self.fine_tune_layers = fine_tune_layers
            # Pop the last fully connected layer of resnet
            self.in_features = self.resnet.fc.in_features # This shows the number of outputs of resnet after amputating it
            # Strip off the last layer
            self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

            # self.fc1 = nn.Linear(self.in_features, int(self.in_features / 2))
            # self.fc2 = nn.Linear(int(self.in_features / 2), int(self.in_features / 4))
            # self.fc3 = nn.Linear(int(self.in_features / 4), int(self.in_features / 8))
            # self.fc4 = nn.Linear(int(self.in_features / 8), int(self.in_features / 16))

            # Make Resnet a fixed feature extractor
            for param in self.resnet.parameters():
                param.requires_grad = False

            # Activate fine-tunning of certain layers if Requested
            if self.fine_tune == True:
                for name, param in self.resnet.named_parameters():
                    if (param.requires_grad == False):
                        for layer in self.fine_tune_layers:
                            if name.startswith(str(layer)):
                                param.requires_grad = True
            # for name, param in self.resnet.named_parameters():
            #     if param.requires_grad == True:
            #         print(name)


        elif self.DeepNetOption == 'LeNet':
            self.lenet = LeNet()

        # Load the EBFDD network that would be loaded
        self.rbfdd = RBFDD(h, pre_rbfdd_input_dim)





    def forward(self, x):
        if self.DeepNetOption == 'ResNet':
            x = self.resnet(x)
            x = torch.squeeze(x)
            # x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            # x = F.relu(self.fc3(x))
            # x = self.fc4(x)

        elif self.DeepNetOption == 'DeepCNN':
            x = self.deepcnn(x)
        elif self.DeepNetOption == 'LeNet':
            x = self.lenet(x)

        # APPLY Lecun's non-linearity here
        x = 1.7159 * torch.tanh(float(2 / 3) * x)
        [x, div] = self.rbfdd(x)
        return x, div

    def partial_forward(self, x):
        if self.DeepNetOption == 'ResNet':
            x = self.resnet(x)
            x = torch.squeeze(x)
            # x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            # x = F.relu(self.fc3(x))
            # x = self.fc4(x)

        elif self.DeepNetOption == 'DeepCNN':
            x = self.deepcnn(x)

        elif self.DeepNetOption == 'LeNet':
            x = self.lenet(x)

        # APPLY Lecun's non-linearity here
        x = 1.7159 * torch.tanh(float(2 / 3) * x)
        return x
