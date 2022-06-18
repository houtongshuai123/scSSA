import torch
import torch.nn as nn
from torch.nn import functional as F

class SSA(nn.Module):
    """
    Defines a semi-supervised autoencoder based on the deep count autoencoder,
    which is essentially an autoencoder with the output space defined to be the
    zero-inflated negative binomial distribution.
    """

    def __init__(self, z_dim, input_dim , n_classes):
        super(SSA, self).__init__()

        #### Network parameters ####
        self.z_dim = z_dim  # dimension of the encoded layer
        self.input_dim = input_dim  # dimension of the input layer
        self.n_classes = n_classes  # number of classes, for using in softmax regression
        h_dim = 800  # dimension of the hidden layer before the encoded layer

        #### Loss parameters ####
        # self.lambd = lambd
        # self.phi = phi
        # self.gamma = gamma

        #### Encoder ####
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        #### Decoder ####
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4_mu = nn.Linear(h_dim, input_dim)  # mu layer
        self.fc4_theta = nn.Linear(h_dim, input_dim)  # theta layer
        self.fc4_pi = nn.Linear(h_dim, input_dim)  # pi layer (logits)

        #### Softmax Regression ####     softmax layer
        self.fc3_dropout = nn.Dropout(p=0.5)
        self.fc3_predict = nn.Linear(z_dim, self.n_classes)
        self.NLLLoss = nn.NLLLoss()

    def encode(self, x):
        return F.elu(self.fc2(F.elu(self.fc1(x))))

    def decode(self, z):
        h3 = F.elu(self.fc3(z))  # the hidden layer after the encoded layer
        hh = self.fc4_mu(h3)
        h4_mu = torch.exp(hh)
        h4_theta = torch.exp(self.fc4_theta(h3))
        h4_pi = self.fc4_pi(h3)  # logits
        return hh, h4_mu, h4_theta, h4_pi

    def forward(self, x):
        z = self.encode(x)
        hh,mu,theta,pi = self.decode(z)
        pred = F.log_softmax(self.fc3_predict(z), dim=-1)
        return  z,hh , mu, theta, pi ,pred


    def calc_losses(self,x, labels,index_list):
        z,hh , mu, theta, pi ,pred= self.forward(x)
        pred_1 = pred[index_list,:]
        nll = nn.NLLLoss(pred_1, labels)
        # pred_lab = torch.max(pred, dim=1)[1]
        reconsLoss = -log_zinb_positive(x, mu, theta, pi).mean()
        totalLoss = reconsLoss + nll
        return  totalLoss
