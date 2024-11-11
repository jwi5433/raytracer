import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Vec3:
    x: float
    y: float
    z: float


    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def normalize(self):
        mag = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        if mag == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x/mag, self.y/mag, self.z/mag)
