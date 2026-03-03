import torch 
from torch import nn
import matplotlib.pyplot as plt

weight =0.7
bias= 0.3

X=torch.arange(0,1,0.02).unsqueeze(dim=1)
Y= weight*X + bias

#splitting the data into the testing the training 
train_split=int(0.8 *len(X))
X_train,Y_train=X[:train_split],Y[:train_split]
X_test,Y_test=X[train_split:],Y[train_split:]

#plotting the data 
def plotting(
    train_data=X_train,
    train_labels=Y_train,
    test_data=X_test,
    test_labels=Y_test,
    predictions=None 
       
):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_labels,c='b',s=4,label='traing data')
    plt.scatter(test_data,test_labels,c='g',s=4,label='testing data')
    plt.legend(loc='best')
    plt.grid()
    if predictions is not None:
        plt.scatter(test_data,predictions,c='b',s=4,label='Prediction by thr model')
        plt.grid()
        plt.legend(loc='best')
        plt.show()
    plt.show()

class linaerregressionmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(1,
            requires_grad=True,
            dtype=torch.float
        ))
        self.bias=nn.Parameter(torch.randn(1,
                            requires_grad=True,
                            dtype=torch.float          
        ))
    def forward(self,X: torch.tensor)-> torch.tensor:
            return self.weight *X + self.bias

torch.manual_seed(42)
model_0=linaerregressionmodel()

print(list(model_0.parameters()))
#named parameters
print(list(model_0.named_parameters()))

#to see the predictions we usr the torch.infrenece mode
with torch.inference_mode():
    y_pred=model_0(X_test)
print(y_pred)
print(Y_test)

plotting(predictions=y_pred)

loss_fn=nn.L1Loss()
optimizer=torch.optim(params=model_0.parameters(),
                      lr=0.01)

epoch=1

torch.manual_seed(42)
for i in range(epoch):
  model_0.train()
  y_pred=model_0(X_train)
  loss=loss_fn(y_pred,Y_train)
  print(f"loss is : {loss}")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  model_0.eval() # turn off the gradient  

with torch.inference_mode():
  y_pred_new=model_0(X_test)
  plotting(y_pred_new)