# Python Raytracer

A flexible raytracer implementation in Python that supports various features including textures, multiple light sources, reflections, and different camera projections.

## Features

- Geometric primitives: spheres, planes, and triangles
- Multiple light types: directional (sun) and point lights (bulbs)
- Material properties including textures and shininess/reflections
- Camera options:
  - Standard perspective projection
  - Fisheye projection
  - Panoramic projection
- Linear color space handling with exposure control
- Support for PNG output with alpha channel

## Installation

### Prerequisites

The raytracer requires the following Python packages:

```bash
pip install numpy Pillow
```

### Usage

1. Create a scene file (e.g., `scene.txt`) with your scene description
2. Run the raytracer:

```bash
python raytracer.py scene.txt
```

## Scene File Format

The scene file uses a simple text format with one command per line. Here are the available commands:

### Basic Setup

```
png <width> <height> <output_filename>  # Set image dimensions and output file
expose <value>                         # Set exposure value for tone mapping
bounces <n>                           # Set maximum number of reflection bounces
```

### Camera Setup

```
eye <x> <y> <z>                       # Set camera position
forward <x> <y> <z>                   # Set camera direction
up <x> <y> <z>                        # Set camera up vector
fisheye                               # Enable fisheye projection
panorama                              # Enable panoramic projection
```

### Materials and Textures

```
color <r> <g> <b>                     # Set current color (RGB values 0-1)
texture <filename>                     # Set current texture (or 'none' to disable)
shininess <s>                         # Set uniform shininess
shininess <r> <g> <b>                 # Set per-channel shininess
```

### Geometry

```
sphere <x> <y> <z> <radius>           # Add sphere
plane <a> <b> <c> <d>                 # Add plane (ax + by + cz + d = 0)
texcoord <u> <v>                      # Set current texture coordinates
xyz <x> <y> <z>                       # Define vertex position
tri <v1> <v2> <v3>                    # Add triangle (vertex indices starting at 1)
```

### Lighting

```
sun <x> <y> <z>                       # Add directional light
bulb <x> <y> <z>                      # Add point light
```

## Example Scene

Here's a simple example scene file:

```
png 800 600 output.png
eye 0 0 5
forward 0 0 -1
up 0 1 0

# Add a red sphere
color 1 0 0
sphere 0 0 0 1

# Add a point light
color 1 1 1
bulb 5 5 5
```

## Advanced Features

### Texturing

The raytracer supports PNG textures with alpha channels. Textures are automatically mapped:
- Spherically for spheres
- Using UV coordinates for triangles

### Reflections

Use the `shininess` command to control surface reflectivity. Higher values create more mirror-like surfaces.

### Multiple Light Sources

Combine `sun` and `bulb` lights for complex lighting:
- `sun` lights are directional with constant intensity
- `bulb` lights attenuate with distance (inverse square law)

