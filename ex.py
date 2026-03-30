import open3d as o3d
import numpy as np
from PIL import Image

# Load skull
skull = o3d.io.read_triangle_mesh("craneo.OBJ")
skull.compute_vertex_normals()

# Load texture using PIL
tex_img = np.asarray(Image.open("texture.jpg").convert("RGB"))

# Get UV coordinates
uvs = np.asarray(skull.triangle_uvs)  # shape: (N*3, 2)
triangles = np.asarray(skull.triangles)

h, w, _ = tex_img.shape
vertex_colors = np.zeros((len(skull.vertices), 3))
vertex_counts = np.zeros(len(skull.vertices))

# Map each UV to a color from texture
tri_uvs = uvs.reshape(-1, 3, 2)  # (num_triangles, 3 vertices, 2 uv)

for i, tri in enumerate(triangles):
    for j, vert_idx in enumerate(tri):
        u, v = tri_uvs[i][j]
        # Flip V axis (OBJ UVs are bottom-up)
        px = int(u * (w - 1)) % w
        py = int((1 - v) * (h - 1)) % h
        color = tex_img[py, px] / 255.0
        vertex_colors[vert_idx] += color
        vertex_counts[vert_idx] += 1

# Average colors for shared vertices
vertex_counts[vertex_counts == 0] = 1
vertex_colors /= vertex_counts[:, np.newaxis]

skull.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

# Visualize
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Skull Viewer", width=1280, height=720)
vis.add_geometry(skull)

ctr = vis.get_view_control()
ctr.set_zoom(0.8)

while True:
    vis.poll_events()
    vis.update_renderer()