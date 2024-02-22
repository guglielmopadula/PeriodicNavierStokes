import numpy as np
import torch
import os
from torch.utils.data import TensorDataset


class PeriodicNavierStokes():
    def __init__(self, batch_size):
        from numpy.random import Generator, PCG64
        self.batch_size = batch_size
        self.data_directory = os.path.join(os.path.dirname(__file__), 'data')
        self.points=torch.tensor(np.load(os.path.join(self.data_directory, 'space.npy')),dtype=torch.float32)
        self.time=torch.tensor(np.load(os.path.join(self.data_directory, 'time.npy')),dtype=torch.float32)
        self.points_red=torch.tensor(np.load(os.path.join(self.data_directory, 'space_red.npy')),dtype=torch.float32)
        self.time_red=torch.tensor(np.load(os.path.join(self.data_directory, 'time_red.npy')),dtype=torch.float32)

        self.A_train=torch.tensor(np.load(os.path.join(self.data_directory, 'a.npy')),dtype=torch.float32)
        self.U_train=torch.tensor(np.load(os.path.join(self.data_directory, 'u.npy')),dtype=torch.float32)
        self.A_super=torch.tensor(np.load(os.path.join(self.data_directory, 'a_super.npy')),dtype=torch.float32)
        self.U_super=torch.tensor(np.load(os.path.join(self.data_directory, 'u_super.npy')),dtype=torch.float32)
        self.A_test=torch.tensor(np.load(os.path.join(self.data_directory, 'a_test.npy')),dtype=torch.float32)
        self.U_test=torch.tensor(np.load(os.path.join(self.data_directory, 'u_test.npy')),dtype=torch.float32)
        self.train_dataset=TensorDataset(self.A_train,self.U_train)
        self.test_dataset=TensorDataset(self.A_test,self.U_test)
        self.train_loader=torch.utils.data.DataLoader(self.train_dataset,batch_size=batch_size,shuffle=False)
        self.test_loader=torch.utils.data.DataLoader(self.test_dataset,batch_size=1,shuffle=False)

