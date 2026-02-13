#rule 1 for the matrix multily the innerr dimentions must be match 
import torch

# tensor=torch.rand(3,2)@torch.rand(3,2)
# print(tensor)
#this gives the error the inner dimentions cannotmatched  3x2 @ 3x2

tensor=torch.rand(3,2)@ torch.rand(2,3)
print(tensor)
#rules 2 
# the result matrix is the shape of the outer matrix

print('3x2 @ 2x3',torch.rand(3,2)@ torch.rand(2,3)) # gives 3 by 3 
print('2x3 @ 3x2',torch.rand(2,3)@ torch.rand(3,2))
print('2x10 @ 10X2',torch.rand(2,10)@ torch.rand(10,2) )
print('10x2 @ 2X10',torch.rand(10,2)@ torch.rand(2,10) )