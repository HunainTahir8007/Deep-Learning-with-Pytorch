import torch 

tensor=torch.arange(1,10)
print(tensor)
#for reshaping
re=tensor.reshape((3,3))
print(re)
# for view
z=tensor.view(1,9)
print(z)
print(re)
#now change the value of the element for view 
z[:,1]=10
print(z)# when we change the z view it also chage the value of the original tensor (Due to same memory location)
print(re)