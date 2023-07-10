import torch
import torch.nn as nn
import torch.nn.functional as F

class denoising_model(nn.Module):
  def __init__(self):
    super(denoising_model,self).__init__()
    self.encoder=nn.Sequential(
                  nn.Linear(4,64),
                  nn.ReLU(True),
                #   nn.Linear(32,64),
                #   nn.ReLU(True),
                #   nn.Linear(64,128),
                #   nn.ReLU(True)
        
                  )
    
    self.decoder=nn.Sequential(
                #   nn.Linear(128,64),
                #   nn.ReLU(True),
                #   nn.Linear(64,32),
                #   nn.ReLU(True),
                  nn.Linear(64,4),
                  nn.ReLU(True)
                #   nn.Sigmoid(),
                  )
    
 
  def forward(self,x):
    x=self.encoder(x)
    x=self.decoder(x)
    
    return x