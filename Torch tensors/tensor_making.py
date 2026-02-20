import torch as tc
import numpy as np

data=[[1,2],[3,4]]

x_data=tc.tensor(data)
print('Data after',x_data)

#--------------tensors from numpy---------------
dat=np.array(data)
arr_data=tc.from_numpy(dat)
print(arr_data)