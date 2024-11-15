import numpy as np
from PIL import Image
import os
import sys

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)
        self.direction = self.direction / np.linalg.norm(direction)

class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)

class Light:
    def __init__(self, direction, color):
        self.direction = np.array(direction)
        self.direction = self.direction / np.linalg.norm(direction)
        self.color = np.array(color)

class Scene:
    def __init__(self):
        self.spheres = []
        self.lights = []
        self.current_color = np.array([1.0, 1.0, 1.0])

class Raytracer:
    def __init__(self):
        self.scene = Scene()
        self.width = 0
        self.height = 0
        self.eye = np.array([0.0, 0.0, 0.0])
        self.forward = np.array([0.0, 0.0, -1.0])
        self.target_up = np.array([0.0, 1.0, 0.0])
        self.right = np.array([1.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.exposure = None
        self.is_fisheye = False
        self.is_panorama = False

    def update_camera_vectors(self):
        """Update right and up vectors based on forward and target_up."""
        # Don't normalize forward - its length affects FOV
        
        # r⃗ = normalized(f⃗ × up)
        self.right = np.cross(self.forward, self.target_up)
        self.right = self.right / np.linalg.norm(self.right)
        
        # u⃗ = normalized(r⃗ × f⃗)
        self.up = np.cross(self.right, self.forward)
        self.up = self.up / np.linalg.norm(self.up)

    def parse_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            command = parts[0]
            
            if command == 'png':
                self.width = int(parts[1])
                self.height = int(parts[2])
                self.output_file = os.path.join('output', parts[3])
            elif command == 'expose':
                self.exposure = float(parts[1])
            elif command == 'eye':
                self.eye = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            elif command == 'forward':
                self.forward = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                self.update_camera_vectors()
            elif command == 'up':
                self.target_up = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                self.target_up = self.target_up / np.linalg.norm(self.target_up)
                self.update_camera_vectors()
            elif command == 'fisheye':
                self.is_fisheye = True
                self.is_panorama = False
            elif command == 'panorama':
                self.is_panorama = True
                self.is_fisheye = False
            elif command == 'color':
                self.scene.current_color = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            elif command == 'sphere':
                center = [float(parts[1]), float(parts[2]), float(parts[3])]
                radius = float(parts[4])
                self.scene.spheres.append(Sphere(center, radius, self.scene.current_color.copy()))
            elif command == 'sun':
                direction = [float(parts[1]), float(parts[2]), float(parts[3])]
                self.scene.lights.append(Light(direction, self.scene.current_color.copy()))

    def intersect_sphere(self, ray, sphere):
        oc = ray.origin - sphere.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - sphere.radius * sphere.radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
            
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        if t < 0:
            t = (-b + np.sqrt(discriminant)) / (2.0 * a)
        if t < 0:
            return None
            
        return t

    def get_sphere_normal(self, point, sphere):
        normal = point - sphere.center
        return normal / np.linalg.norm(normal)

    def cast_ray(self, ray):
        closest_t = float('inf')
        hit_sphere = None
        
        for sphere in self.scene.spheres:
            t = self.intersect_sphere(ray, sphere)
            if t is not None and t < closest_t:
                closest_t = t
                hit_sphere = sphere
        
        if hit_sphere is None:
            return np.array([0.0, 0.0, 0.0, 0.0])
            
        hit_point = ray.origin + closest_t * ray.direction
        normal = self.get_sphere_normal(hit_point, hit_sphere)
        
        if np.dot(normal, ray.direction) > 0:
            normal = -normal
            
        if not self.scene.lights:
            return np.array([0.0, 0.0, 0.0, 1.0])
            
        color = np.zeros(3)
        for light in self.scene.lights:
            shadow_ray = Ray(hit_point + normal * 0.001, light.direction)
            in_shadow = False
            
            for sphere in self.scene.spheres:
                t = self.intersect_sphere(shadow_ray, sphere)
                if t is not None:
                    in_shadow = True
                    break
            
            if not in_shadow:
                diffuse = max(0, np.dot(normal, light.direction))
                color += hit_sphere.color * light.color * diffuse
        
        return np.append(np.clip(color, 0, 1), 1.0)

    def apply_exposure(self, color):
        if self.exposure is None:
            return color
        
        rgb = color[:3]
        alpha = color[3:]
        exposed = 1.0 - np.exp(-self.exposure * rgb)
        return np.concatenate([exposed, alpha])

    def linear_to_srgb(self, linear):
        rgb = linear[:3]
        alpha = linear[3:]
        srgb = np.where(rgb <= 0.0031308,
                       rgb * 12.92,
                       1.055 * np.power(rgb, 1/2.4) - 0.055)
        return np.concatenate([srgb, alpha])

    def get_ray_direction(self, x, y):
        sx = (2.0 * x - self.width) / max(self.width, self.height)
        sy = (self.height - 2.0 * y) / max(self.width, self.height)
    
        if self.is_fisheye:
            r2 = sx * sx + sy * sy
            # Check if point is within the fisheye circle
            if r2 >= 1:
                return None
            
            # Only compute sqrt for points inside the circle
            z = np.sqrt(1.0 - r2)
            direction = z * self.forward + sx * self.right + sy * self.up
        
        elif self.is_panorama:
            # Convert x coordinate to longitude (0 to 2π)
            longitude = (x / self.width) * 2.0 * np.pi
            # Convert y coordinate to latitude (-π/2 to π/2)
            latitude = ((self.height - y) / self.height - 0.5) * np.pi
        
            # Convert spherical coordinates to Cartesian
            x = np.cos(latitude) * np.sin(longitude)
            y = np.sin(latitude)
            z = np.cos(latitude) * np.cos(longitude)
        
            # Transform direction based on camera orientation
            direction = (z * self.forward + 
                        x * self.right + 
                        y * self.up)
        else:
            direction = self.forward + sx * self.right + sy * self.up
        
        return direction / np.linalg.norm(direction)

    def render(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        image = np.zeros((self.height, self.width, 4))
        
        for y in range(self.height):
            for x in range(self.width):
                direction = self.get_ray_direction(x, y)
                
                if direction is None:  # Outside fisheye circle
                    image[y, x] = [0, 0, 0, 0]  # Transparent
                    continue
                    
                ray = Ray(self.eye, direction)
                
                color = self.cast_ray(ray)
                color = self.apply_exposure(color)
                color = self.linear_to_srgb(color)
                image[y, x] = color
        
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(image, 'RGBA').save(self.output_file, 'PNG')

def main(input_file):
    raytracer = Raytracer()
    raytracer.parse_file(input_file)
    raytracer.render()

if __name__ == "__main__":
    main(sys.argv[1])