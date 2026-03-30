import trimesh
from PIL import Image
import numpy as np

mesh = trimesh.load("heart.glb")
heart_geom = mesh.geometry['GLTF']
mat = heart_geom.visual.material

print("PBR Material attributes:")
print("  baseColorTexture:", mat.baseColorTexture)
print("  metallicRoughnessTexture:", mat.metallicRoughnessTexture)
print("  normalTexture:", mat.normalTexture)
print("  emissiveTexture:", mat.emissiveTexture)
print("  baseColorFactor:", mat.baseColorFactor)

# Save whichever textures exist
if mat.baseColorTexture is not None:
    mat.baseColorTexture.save("heart_basecolor.jpg")
    print("Saved heart_basecolor.jpg!")

if mat.normalTexture is not None:
    mat.normalTexture.save("heart_normal.jpg")
    print("Saved heart_normal.jpg!")

if mat.emissiveTexture is not None:
    mat.emissiveTexture.save("heart_emissive.jpg")
    print("Saved heart_emissive.jpg!")

if mat.metallicRoughnessTexture is not None:
    mat.metallicRoughnessTexture.save("heart_metallic.jpg")
    print("Saved heart_metallic.jpg!")