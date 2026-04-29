import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np
import threading
from pynput import keyboard
from PIL import Image
import os

# --- CONFIGURATION ---
SKULL_PATH   = "craneo.obj"
HEART_PATH   = "heart.obj"
HEART_TEX    = "heart.png"   # change to your actual texture filename
SMOOTHING_FACTOR = 0.20

# --- STATE ---
is_locked   = False
should_quit = False
state_lock  = threading.Lock()
active_model = "skull"   # "skull" or "heart"

def on_press(key):
    global is_locked, should_quit, active_model
    try:
        with state_lock:
            if key.char == 'q':
                should_quit = True
            elif key.char == 'l':
                is_locked = not is_locked
            elif key.char == 's':
                active_model = "skull"
                print("[INFO] Switched to Skull")
            elif key.char == 'h':
                active_model = "heart"
                print("[INFO] Switched to Heart")
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

# --- TEXTURE LOADER ---
def apply_texture(mesh, tex_path):
    """Sample UV texture and bake into vertex colours."""
    print(f"[INFO] Applying texture from {tex_path}...")
    tex_img = np.asarray(Image.open(tex_path).convert("RGB"))
    uvs      = np.asarray(mesh.triangle_uvs)
    triangles = np.asarray(mesh.triangles)
    h, w, _  = tex_img.shape

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
    print("[INFO] Texture applied.")
    return mesh

# --- LOAD SKULL (cyan hologram style) ---
print("[INFO] Loading skull...")
try:
    skull = o3d.io.read_triangle_mesh(SKULL_PATH)
    skull.compute_vertex_normals()
    skull.paint_uniform_color([0.9, 0.85, 0.7])   # cyan hologram
except Exception as e:
    print(f"[WARN] {e} — using fallback sphere for skull")
    skull = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=60)
    skull.paint_uniform_color([0.9, 0.85, 0.7])

skull.translate(-skull.get_center())

# --- LOAD HEART (with texture) ---
print("[INFO] Loading heart...")
try:
    heart = o3d.io.read_triangle_mesh(HEART_PATH)
    heart.compute_vertex_normals()

    if os.path.exists(HEART_TEX) and heart.has_triangle_uvs():
        heart = apply_texture(heart, HEART_TEX)
    else:
        print("[WARN] Heart texture not found or no UVs — using red fallback")
        heart.paint_uniform_color([0.85, 0.1, 0.1])   # red fallback
except Exception as e:
    print(f"[WARN] {e} — using fallback sphere for heart")
    heart = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=60)
    heart.paint_uniform_color([0.85, 0.1, 0.1])
skull_extent = np.max(skull.get_axis_aligned_bounding_box().get_extent())
heart_extent = np.max(heart.get_axis_aligned_bounding_box().get_extent())
heart.scale(skull_extent / heart_extent, center=heart.get_center())
heart.translate(-heart.get_center())

# --- OPEN3D WINDOW ---
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="HoloGesture", width=335, height=335, visible=False)

opt = vis.get_render_option()
opt.background_color = np.array([0, 0, 0])
opt.light_on = True

# Start with skull
current_mesh = skull
vis.add_geometry(current_mesh)
last_model = "skull"

# --- HAND TRACKING ---
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(min_detection_confidence=0.8,
                             min_tracking_confidence=0.8)
cap = cv2.VideoCapture(0)

# --- MOVEMENT STATE ---
curr_rot     = np.array([0.0, 0.0, 0.0])
target_rot   = np.array([0.0, 0.0, 0.0])
curr_trans   = np.array([0.0, 0.0, 0.0])
target_trans = np.array([0.0, 0.0, 0.0])
prev_hand_pos   = None
prev_pinch_dist = None

# --- GESTURE HELPERS ---
def is_fist(hand_landmarks):
    tips = [8, 12, 16, 20]
    mcps = [5,  9, 13, 17]
    return all(
        hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y
        for tip, mcp in zip(tips, mcps)
    )

def is_pinching(hand_landmarks):
    idx   = np.array([hand_landmarks.landmark[8].x,
                      hand_landmarks.landmark[8].y])
    thumb = np.array([hand_landmarks.landmark[4].x,
                      hand_landmarks.landmark[4].y])
    return np.linalg.norm(idx - thumb) < 0.05

# --- 4-VIEW CANVAS ---
def make_4view_canvas(frame_bgr):
    h, w = frame_bgr.shape[:2]
    S = min(h, w)
    f = frame_bgr[:S, :S]
    
    front = cv2.flip(f, 0)
    back  = cv2.flip(f, 1)
    left  = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
    right = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)

    canvas = np.zeros((S * 3, S * 3, 3), dtype=np.uint8)
    canvas[0:S,     S:S*2]   = front
    canvas[S:S*2,   0:S]     = left
    canvas[S:S*2,   S*2:S*3] = right
    canvas[S*2:S*3, S:S*2]   = back
    return canvas

# --- RESET TRANSFORM HELPER ---
def reset_transform():
    global curr_rot, target_rot, curr_trans, target_trans
    curr_rot     = np.array([0.0, 0.0, 0.0])
    target_rot   = np.array([0.0, 0.0, 0.0])
    curr_trans   = np.array([0.0, 0.0, 0.0])
    target_trans = np.array([0.0, 0.0, 0.0])

print("\nControls:")
print("  S key     → Switch to Skull")
print("  H key     → Switch to Heart")
print("  Move hand → Rotate")
print("  Pinch     → Drag")
print("  2 Hands   → Zoom")
print("  Fist      → Freeze / Unfreeze")
print("  L key     → Lock / Unlock")
print("  Q key     → Quit\n")

# --- MAIN LOOP ---
while cap.isOpened():
    with state_lock:
        if should_quit:
            break
        locked = is_locked
        model  = active_model

    # --- SWITCH MODEL IF NEEDED ---
    if model != last_model:
        vis.remove_geometry(current_mesh, reset_bounding_box=False)
        current_mesh = skull if model == "skull" else heart
        # Re-centre after switching
        current_mesh.translate(-current_mesh.get_center())
        vis.add_geometry(current_mesh, reset_bounding_box=True)
        reset_transform()
        last_model = model
        print(f"[INFO] Now showing: {model}")

    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(rgb_frame)

    frozen = False

    if results.multi_hand_landmarks and not locked:
        for lm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        hand = results.multi_hand_landmarks[0]

        if is_fist(hand):
            frozen = True
            prev_hand_pos = np.array([hand.landmark[8].x,
                                      hand.landmark[8].y])
            cv2.putText(frame, "FROZEN", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        else:
            idx_tip  = np.array([hand.landmark[8].x, hand.landmark[8].y])
            pinching = is_pinching(hand)

            if prev_hand_pos is not None:
                delta = idx_tip - prev_hand_pos
                if pinching:
                    target_trans[0] += delta[0] * 5
                    target_trans[1] -= delta[1] * 5
                else:
                    target_rot[0] += delta[1] * 10
                    target_rot[1] += delta[0] * 10

            if len(results.multi_hand_landmarks) > 1:
                h1 = np.array([results.multi_hand_landmarks[0].landmark[8].x,
                                results.multi_hand_landmarks[0].landmark[8].y])
                h2 = np.array([results.multi_hand_landmarks[1].landmark[8].x,
                                results.multi_hand_landmarks[1].landmark[8].y])
                curr_dist = np.linalg.norm(h1 - h2)
                if prev_pinch_dist is not None:
                    zoom = 1 + (curr_dist - prev_pinch_dist) * 2
                    current_mesh.scale(zoom, center=current_mesh.get_center())
                prev_pinch_dist = curr_dist
            else:
                prev_pinch_dist = None

            prev_hand_pos = idx_tip
    else:
        prev_hand_pos   = None
        prev_pinch_dist = None

    # --- SMOOTH MOVEMENT ---
    if not frozen and not locked:
        step_rot = (target_rot - curr_rot) * SMOOTHING_FACTOR
        if np.linalg.norm(step_rot) > 0.001:
            R = current_mesh.get_rotation_matrix_from_xyz(step_rot)
            current_mesh.rotate(R, center=current_mesh.get_center())
            curr_rot += step_rot

        step_trans = (target_trans - curr_trans) * SMOOTHING_FACTOR
        if np.linalg.norm(step_trans) > 0.001:
            current_mesh.translate(step_trans)
            curr_trans += step_trans

    # --- RENDER ---
    vis.update_geometry(current_mesh)
    vis.poll_events()
    vis.update_renderer()

    raw      = vis.capture_screen_float_buffer(do_render=True)
    rendered = (np.asarray(raw)[:, :, ::-1] * 255).astype(np.uint8)

    # --- 4-VIEW CANVAS ---
    canvas = make_4view_canvas(rendered)

    # Overlay: model name + status
    model_label = "SKULL" if model == "skull" else "HEART"
    status      = "LOCKED" if locked else ("FROZEN" if frozen else "ACTIVE")
    color       = (0,255,80) if locked else ((0,165,255) if frozen else (0,200,255))
    cv2.putText(canvas, f"HoloGesture  |  {model_label}  |  {status}",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Driver view overlay
    cv2.putText(frame, f"Model: {model_label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, status, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("HoloGesture — 4-View Pepper's Ghost", canvas)
    cv2.imshow("Driver View", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
print("[INFO] Session ended.")