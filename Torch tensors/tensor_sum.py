import torch 

tensor=torch.ones(4,4)
agg=tensor.sum()
print(agg)
agg_item=agg.item()
print(agg_item, type(agg_item))
print('--------------------------------')
print(tensor)
tensor.add_(5)
print(tensor)