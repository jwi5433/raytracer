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

class Plane:
    def __init__(self, a, b, c, d, color):
        self.normal = np.array([a, b, c])
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.d = d
        self.color = np.array(color)

class Triangle:
    def __init__(self, v1, v2, v3, color):
        self.vertices = [np.array(v1), np.array(v2), np.array(v3)]
        self.color = np.array(color)
        # Calculate normal
        edge1 = self.vertices[1] - self.vertices[0]
        edge2 = self.vertices[2] - self.vertices[0]
        self.normal = np.cross(edge1, edge2)
        self.normal = self.normal / np.linalg.norm(self.normal)

class Light:
    def __init__(self, direction, color):
        self.direction = np.array(direction)
        self.direction = self.direction / np.linalg.norm(direction)
        self.color = np.array(color)

class Scene:
    def __init__(self):
        self.spheres = []
        self.planes = []
        self.triangles = []
        self.vertices = []
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
        self.right = np.cross(self.forward, self.target_up)
        self.right = self.right / np.linalg.norm(self.right)
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
            elif command == 'plane':
                a, b, c, d = [float(p) for p in parts[1:]]
                self.scene.planes.append(Plane(a, b, c, d, self.scene.current_color.copy()))
            elif command == 'xyz':
                x, y, z = [float(p) for p in parts[1:4]]
                self.scene.vertices.append(np.array([x, y, z]))
            elif command == 'tri':
                indices = [int(i) - 1 for i in parts[1:4]]
                indices = [i if i >= 0 else len(self.scene.vertices) + i for i in indices]
                v1 = self.scene.vertices[indices[0]]
                v2 = self.scene.vertices[indices[1]]
                v3 = self.scene.vertices[indices[2]]
                self.scene.triangles.append(Triangle(v1, v2, v3, self.scene.current_color.copy()))
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

    def intersect_plane(self, ray, plane):
        denom = np.dot(plane.normal, ray.direction)
        if abs(denom) < 1e-6:
            return None
        t = -(np.dot(plane.normal, ray.origin) + plane.d) / denom
        if t < 0:
            return None
        return t

    def intersect_triangle(self, ray, triangle):
        v0, v1, v2 = triangle.vertices
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray.direction, edge2)
        a = np.dot(edge1, h)
        
        if abs(a) < 1e-6:
            return None
            
        f = 1.0 / a
        s = ray.origin - v0
        u = f * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            return None
            
        q = np.cross(s, edge1)
        v = f * np.dot(ray.direction, q)
        
        if v < 0.0 or u + v > 1.0:
            return None
            
        t = f * np.dot(edge2, q)
        
        if t < 0:
            return None
            
        return t

    def get_sphere_normal(self, point, sphere):
        normal = point - sphere.center
        return normal / np.linalg.norm(normal)

    def cast_ray(self, ray):
        closest_t = float('inf')
        hit_object = None
        object_type = None
        
        # Check sphere intersections
        for sphere in self.scene.spheres:
            t = self.intersect_sphere(ray, sphere)
            if t is not None and t < closest_t:
                closest_t = t
                hit_object = sphere
                object_type = 'sphere'

        # Check plane intersections
        for plane in self.scene.planes:
            t = self.intersect_plane(ray, plane)
            if t is not None and t < closest_t:
                closest_t = t
                hit_object = plane
                object_type = 'plane'

        # Check triangle intersections
        for triangle in self.scene.triangles:
            t = self.intersect_triangle(ray, triangle)
            if t is not None and t < closest_t:
                closest_t = t
                hit_object = triangle
                object_type = 'triangle'
        
        if hit_object is None:
            return np.array([0.0, 0.0, 0.0, 0.0])
            
        hit_point = ray.origin + closest_t * ray.direction
        
        # Get normal based on object type
        if object_type == 'sphere':
            normal = self.get_sphere_normal(hit_point, hit_object)
        elif object_type == 'plane':
            normal = hit_object.normal
        else:  # triangle
            normal = hit_object.normal
        
        if np.dot(normal, ray.direction) > 0:
            normal = -normal
            
        if not self.scene.lights:
            return np.array([0.0, 0.0, 0.0, 1.0])
            
        color = np.zeros(3)
        for light in self.scene.lights:
            shadow_ray = Ray(hit_point + normal * 0.001, light.direction)
            in_shadow = False
            
            # Check sphere shadows
            for sphere in self.scene.spheres:
                if self.intersect_sphere(shadow_ray, sphere) is not None:
                    in_shadow = True
                    break
                    
            # Check plane shadows
            if not in_shadow:
                for plane in self.scene.planes:
                    if self.intersect_plane(shadow_ray, plane) is not None:
                        in_shadow = True
                        break
                        
            # Check triangle shadows
            if not in_shadow:
                for triangle in self.scene.triangles:
                    if self.intersect_triangle(shadow_ray, triangle) is not None:
                        in_shadow = True
                        break
            
            if not in_shadow:
                diffuse = max(0, np.dot(normal, light.direction))
                color += hit_object.color * light.color * diffuse
        
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
            if r2 >= 1:
                return None
            z = np.sqrt(1.0 - r2)
            direction = z * self.forward + sx * self.right + sy * self.up
        elif self.is_panorama:
            longitude = (x / self.width) * 2.0 * np.pi
            latitude = ((self.height - y) / self.height - 0.5) * np.pi
            
            x = np.cos(latitude) * np.sin(longitude)
            y = np.sin(latitude)
            z = np.cos(latitude) * np.cos(longitude)
            
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
                
                if direction is None:
                    image[y, x] = [0, 0, 0, 0]
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