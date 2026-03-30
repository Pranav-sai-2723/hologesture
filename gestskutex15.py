import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np
from PIL import Image
import os

# --- CONFIGURATION ---
MODEL_PATH = "craneo.OBJ"
TEXTURE_PATH = "texture.jpg"
SMOOTHING_FACTOR = 0.20
ROTATION_SENSITIVITY = 0.1
ZOOM_SENSITIVITY = 0.009

# --- TEXTURE LOADER FUNCTION ---96
def apply_texture(mesh, tex_path):
    print("Applying texture, please wait...")
    tex_img = np.asarray(Image.open(tex_path).convert("RGB"))
    uvs = np.asarray(mesh.triangle_uvs)
    triangles = np.asarray(mesh.triangles)
    h, w, _ = tex_img.shape
    vertex_colors = np.zeros((len(mesh.vertices), 3))
    vertex_counts = np.zeros(len(mesh.vertices))
    tri_uvs = uvs.reshape(-1, 3, 2)

    for i, tri in enumerate(triangles):
        for j, vert_idx in enumerate(tri):
            u, v = tri_uvs[i][j]
            px = int(u * (w - 1)) % w
            py = int((1 - v) * (h - 1)) % h
            color = tex_img[py, px] / 255.0
            vertex_colors[vert_idx] += color
            vertex_counts[vert_idx] += 1

    vertex_counts[vertex_counts == 0] = 1
    vertex_colors /= vertex_counts[:, np.newaxis]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    print("Texture applied successfully!")
    return mesh

# --- INITIALIZE OPEN3D ---
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Skull", width=1280, height=720)

# --- LOAD SKULL WITH TEXTURE ---
try:
    skull = o3d.io.read_triangle_mesh(MODEL_PATH)
    skull.compute_vertex_normals()

    # Apply texture if file exists
    if os.path.exists(TEXTURE_PATH) and skull.has_triangle_uvs():
        skull = apply_texture(skull, TEXTURE_PATH)
    else:
        print("[WARN] Texture not found or no UVs, using bone color")
        skull.paint_uniform_color([0.9, 0.85, 0.7])

except Exception as e:
    print(f"[WARN] {e} — Using fallback sphere")
    skull = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=100)
    skull.paint_uniform_color([0.7, 0.7, 0.7])

skull.translate(-skull.get_center())
vis.add_geometry(skull)

# --- HAND TRACKING SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
cap = cv2.VideoCapture(0)

# --- STATE VARIABLES ---
curr_rot = np.array([0.0, 0.0, 0.0])
target_rot = np.array([0.0, 0.0, 0.0])
curr_trans = np.array([0.0, 0.0, 0.0])
target_trans = np.array([0.0, 0.0, 0.0])

prev_hand_pos = None
prev_pinch_dist = None
is_locked = False

print("Controls:")
print("- Move Hand : Rotate")
print("- Pinch (Thumb+Index) : Drag")
print("- Two Hands : Zoom")
print("- L : Lock/Unlock")
print("- Q : Quit")

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and not is_locked:

        # Draw hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        hand = results.multi_hand_landmarks[0]
        idx_tip = np.array([hand.landmark[8].x, hand.landmark[8].y])
        thumb_tip = np.array([hand.landmark[4].x, hand.landmark[4].y])

        # Pinch detection
        dist = np.linalg.norm(idx_tip - thumb_tip)
        is_pinching = dist < 0.05

        if prev_hand_pos is not None:
            delta = idx_tip - prev_hand_pos
            if is_pinching:
                target_trans[0] += delta[0] * 5
                target_trans[1] -= delta[1] * 5
            else:
                target_rot[0] += delta[1] * 10
                target_rot[1] += delta[0] * 10

        # Zoom with two hands
        if len(results.multi_hand_landmarks) > 1:
            h1 = np.array([results.multi_hand_landmarks[0].landmark[8].x,
                           results.multi_hand_landmarks[0].landmark[8].y])
            h2 = np.array([results.multi_hand_landmarks[1].landmark[8].x,
                           results.multi_hand_landmarks[1].landmark[8].y])
            curr_dist = np.linalg.norm(h1 - h2)
            if prev_pinch_dist is not None:
                zoom = 1 + (curr_dist - prev_pinch_dist) * 2
                skull.scale(zoom, center=skull.get_center())
            prev_pinch_dist = curr_dist
        else:
            prev_pinch_dist = None

        prev_hand_pos = idx_tip
    else:
        prev_hand_pos = None

    # --- SMOOTH MOVEMENT ---
    step_rot = (target_rot - curr_rot) * SMOOTHING_FACTOR
    if np.linalg.norm(step_rot) > 0.001:
        R = skull.get_rotation_matrix_from_xyz(step_rot)
        skull.rotate(R, center=skull.get_center())
        curr_rot += step_rot

    step_trans = (target_trans - curr_trans) * SMOOTHING_FACTOR
    if np.linalg.norm(step_trans) > 0.001:
        skull.translate(step_trans)
        curr_trans += step_trans

    # Update 3D window
    vis.update_geometry(skull)
    vis.poll_events()
    vis.update_renderer()

    # UI Text
    cv2.putText(frame, "SKULL INTERFACE ACTIVE", (20, 40), 1, 1.5, (0, 0, 255), 2)
    status = "LOCKED" if is_locked else "UNLOCKED"
    cv2.putText(frame, status, (20, 80), 1, 1.5,
                (0, 255, 0) if is_locked else (0, 0, 255), 2)

    cv2.imshow("Driver View", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('l'):
        is_locked = not is_locked

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()