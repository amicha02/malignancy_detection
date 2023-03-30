print ("Always executed")


import torch
from torch.utils.data import Dataset, DataLoader
import argparse

class MyDataset:
    def __init__(self):
        self.samples = [i for i in range(200)]
        
    def __getitem__(self, index):
        return self.samples[index] - 200
    
    def __len__(self):
        return 50 #<1>




#1 If the length of the dataset based on the __len__ method is 100, then you won't be able to access the remaining 100 samples in self.samples beyond the first 100 samples.


my_dataset = MyDataset()
batch_size = 10
num_epochs = 0
num_iterations = len(my_dataset) * num_epochs  # total number of iterations over the dataset
data_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    total_batch_iterations = 0  
    for i, batch in enumerate(data_loader):
        # do some training on the batch
        total_batch_iterations += 1
        print(i)
        print(batch)
    print(total_batch_iterations)
    break



#dataset = MyDataset()
#print(dataset)
#print(len(dataset))
#print(len(list(dataset)))
#first_sample = dataset[120]
#print(first_sample)
#print(len(dataset))





  
        
if __name__ == "__main__":
    print ("Executed when invoked directly")
else:
    print ("Executed when imported")

