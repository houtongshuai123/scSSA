import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from semi_Autoencoder import SSA
from log_zinb_positive import log_zinb_positive


df4 = torch.from_numpy(X_train)
df4= df4.T
df4 = df4.float()
nrows = df4.shape[1]
EPOCH= 150           #Iterations
BATCH_SIZE= 256
LR= 0.0001         #Learning rate


label = y_test.astype(int)
label = torch.tensor(label)
label = label.long()

model = SSA(400, nrows ,6)
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
hh = hh.detach().numpy()
mu = mu.detach().numpy()
theta = theta.detach().numpy()
pi = pi.detach().numpy()



