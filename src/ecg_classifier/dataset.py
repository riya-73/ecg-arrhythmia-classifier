import torch
from torch.utils.data import Dataset
class BeatDataset(Dataset):
    def __init__(self, x, y): self.x=torch.from_numpy(x[:,None,:].astype('float32')); self.y=torch.from_numpy(y.astype('int64'))
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.x[i],self.y[i]
