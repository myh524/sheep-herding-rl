import numpy as np


class RunningMeanStd:
    """Running mean and standard deviation for reward normalization"""
    
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
    
    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x.flatten())
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean.astype(np.float32)
        self.var = new_var.astype(np.float32)
        self.count = tot_count
    
    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
