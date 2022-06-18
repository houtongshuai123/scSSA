import pandas as pd
import numpy as np
from numpy import unique
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.nn import functional as F
import torch

## input data biase
csvFilename = '...\\Data\\datasets\\biase\\biase_data.csv'
data = pd.read_csv(csvFilename,header=0,index_col=0)
## log transformation
X_train = np.log2(data+1)


### import label
csvFilename1 = '...\\Data\\datasets\\biase\\biase_celldata.csv'
true_label = pd.read_csv(csvFilename1,header=0,index_col=0)
true_label[true_label=='2cell']=1
true_label[true_label=='4cell']=2
true_label[true_label=='blast']=3
true_label[true_label=='zygote']=0


X_train = np.array(X_train)
true_label = np.array(true_label)
true_label = true_label.reshape(56,)
x_train , x_test , y_train ,y_test = train_test_split(X_train.T,true_label,test_size=0.2,random_state=5)
n = X_train.T.shape[0]
m = x_test.shape[0]
index_list = []
for i in range(m) :
    for j in range(n):
        pp = (X_train.T[j,:]==x_test[i,:]).all()
        if pp==True:
            index_list.append(j)



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

        #### Softmax Regression ####
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
        reconsLoss = -log_zinb_positive(x, mu, theta, pi).mean()
        totalLoss = reconsLoss
        return  totalLoss

def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    """
    Adopted from: https://github.com/YosefLab/scVI/blob/master/scvi/models/log_likelihood.py#L11
    Equations follow the paper: https://www.nature.com/articles/s41467-017-02554-5

    Parameters
    ----------
    mu: tensor (nsamples, nfeatures)
        Mean of the negative binomial (has to be positive support).
    theta: tensor (nsamples, nfeatures)
        Inverse dispersion parameter (has to be positive support).
    pi: tensor (n_samples, nfeatures)
        Logit of the dropout parameter (real support).
    eps: numeric
        Numerical stability constant.
    """

    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = - pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = - softplus_pi + \
        pi_theta_log + \
        x * (torch.log(mu + eps) - log_theta_mu_eps) + \
        torch.lgamma(x + theta) - \
        torch.lgamma(theta) - \
        torch.lgamma(x + 1)
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return torch.sum(res, dim=-1)



df4 = torch.from_numpy(X_train)
df4= df4.T
df4 = df4.float()
nrows = df4.shape[1]
EPOCH= 150
BATCH_SIZE= 256
LR= 0.0001

y_test = y_test.reshape(y_test.shape[0],)
label = y_test.astype(int)
label = torch.tensor(label)
label = label.long()

model = SSA(400, nrows ,4)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
nllloss = nn.NLLLoss()

for epoch in range(EPOCH):
    z,hh,mu,theta,pi,pred = model(df4)
    pred = pred[index_list, :]
    loss = -log_zinb_positive(df4, mu, theta, pi).mean()/10000 + nllloss(pred,label)
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if epoch % 10 == 0:
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())


z = z.detach().numpy()
from sklearn.decomposition import FastICA
#FastICA
transformer = FastICA(n_components=2,whiten=True)
X_reduction = transformer.fit_transform(z)


from sklearn import mixture
#GaussianMixture clustering
lowest_bic = np.infty
bic = []
n_components_range = range(1, 56)
for n_components in n_components_range:
    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=n_components)
    gmm.fit(X_reduction)
    bic.append(gmm.bic(X_reduction))
    if bic[-1] < lowest_bic:
       lowest_bic = bic[-1]
       best_gmm = gmm
bic = np.array(bic)
clustering = best_gmm
pred_label = clustering.predict(X_reduction)


##evaluation
from sklearn.metrics.cluster import normalized_mutual_info_score
NMI = normalized_mutual_info_score(true_label,pred_label)
print(NMI)
from sklearn.metrics.cluster import adjusted_rand_score
ARI = adjusted_rand_score(true_label,pred_label)
print(ARI)
from sklearn.metrics.cluster import homogeneity_score
HOMO = homogeneity_score(true_label,pred_label)
print(HOMO)
from sklearn.metrics.cluster import completeness_score
COMP = completeness_score(true_label,pred_label)
print(COMP)










