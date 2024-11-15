import numpy as np
from PIL import Image
import os
import sys

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)
        self.direction = self.direction / np.linalg.norm(self.direction)

class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)

class Light:
    def __init__(self, direction, color):
        self.direction = np.array(direction)
        self.direction = self.direction / np.linalg.norm(self.direction)
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
        self.right = np.array([1.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])

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
            
        # If no lights, return black with full opacity
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

    def linear_to_srgb(self, linear):
        rgb = linear[:3]
        alpha = linear[3:]
        srgb = np.where(rgb <= 0.0031308,
                       rgb * 12.92,
                       1.055 * np.power(rgb, 1/2.4) - 0.055)
        return np.concatenate([srgb, alpha])

    def render(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        image = np.zeros((self.height, self.width, 4))
        
        for y in range(self.height):
            for x in range(self.width):
                sx = (2.0 * x - self.width) / max(self.width, self.height)
                sy = (self.height - 2.0 * y) / max(self.width, self.height)
                
                direction = self.forward + sx * self.right + sy * self.up
                direction = direction / np.linalg.norm(direction)
                ray = Ray(self.eye, direction)
                
                color = self.cast_ray(ray)
                image[y, x] = self.linear_to_srgb(color)
        
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(image, 'RGBA').save(self.output_file, 'PNG')

def main(input_file):
    raytracer = Raytracer()
    raytracer.parse_file(input_file)
    raytracer.render()

if __name__ == "__main__":
    main(sys.argv[1])