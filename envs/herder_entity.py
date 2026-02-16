import numpy as np


class HerderEntity:
    """Herder entity with physical constraints for smooth movement"""
    
    def __init__(self, position, max_speed=3.0, max_accel=1.0):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.target_position = None
        self.max_speed = max_speed
        self.max_accel = max_accel
    
    def set_target(self, target_position):
        """Set target position (from high-level policy)"""
        self.target_position = np.array(target_position, dtype=np.float32)
    
    def update(self, dt=0.1):
        """Update position towards target with physical constraints"""
        if self.target_position is None:
            return
        
        direction = self.target_position - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:
            self.velocity *= 0.5
            return
        
        desired_speed = min(self.max_speed, distance / dt)
        desired_velocity = direction / distance * desired_speed
        
        accel = desired_velocity - self.velocity
        accel_mag = np.linalg.norm(accel)
        if accel_mag > self.max_accel * dt:
            accel = accel / accel_mag * self.max_accel * dt
        
        self.velocity += accel
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
        
        self.position += self.velocity * dt
    
    def get_position(self):
        """Get current position"""
        return self.position.copy()
    
    def get_velocity(self):
        """Get current velocity"""
        return self.velocity.copy()
