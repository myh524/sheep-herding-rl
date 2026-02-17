import numpy as np


class RunningMeanStd:
    """Running mean and standard deviation for reward normalization
    
    Features:
    - Exponential moving average for smoother updates
    - Clip extreme values to prevent instability
    - Minimum variance to prevent division by zero
    """
    
    def __init__(self, epsilon=1e-4, shape=(), clip_range=10.0):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.clip_range = clip_range
    
    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        # Skip if contains NaN or Inf
        if np.isnan(x).any() or np.isinf(x).any():
            return
        # Clip extreme values before updating statistics
        x = np.clip(x, -100.0, 100.0)
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x.flatten())
        if np.isnan(batch_mean) or np.isnan(batch_var):
            return
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
        self.var = np.maximum(new_var.astype(np.float32), 1e-4)
        self.count = tot_count
    
    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        if np.isnan(x).any() or np.isinf(x).any():
            return np.zeros_like(x)
        normalized = (x - self.mean) / np.sqrt(self.var + 1e-8)
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)
        return normalized
