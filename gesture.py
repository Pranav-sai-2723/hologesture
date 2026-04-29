import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np
import threading
from pynput import keyboard
from PIL import Image
import os

# --- CONFIGURATION ---
SKULL_PATH = "craneo.obj"
HEART_PATH = "heart.obj"
HEART_TEX  = "heart.png"   # your actual texture filename
SMOOTHING_FACTOR = 0.09

# --- STATE ---
is_locked    = False
should_quit  = False
active_model = "skull"   # "skull" or "heart"
state_lock   = threading.Lock()

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
    print(f"[INFO] Applying texture: {tex_path}")
    tex_img = np.asarray(Image.open(tex_path).convert("RGB"))
    uvs       = np.asarray(mesh.triangle_uvs)
    triangles = np.asarray(mesh.triangles)
    h, w, _   = tex_img.shape
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

# --- LOAD SKULL ---
print("[INFO] Loading skull...")
try:
    skull = o3d.io.read_triangle_mesh(SKULL_PATH)
    skull.compute_vertex_normals()
    if not skull.has_vertex_colors():
        skull.paint_uniform_color([0.9, 0.85, 0.7])  # bone colour
except Exception as e:
    print(f"[WARN] {e} — using fallback sphere for skull")
    skull = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=60)
    skull.paint_uniform_color([0.9, 0.85, 0.7])
skull.translate(-skull.get_center())

# --- LOAD HEART ---
print("[INFO] Loading heart...")
try:
    heart = o3d.io.read_triangle_mesh(HEART_PATH)
    heart.compute_vertex_normals()
    if os.path.exists(HEART_TEX) and heart.has_triangle_uvs():
        heart = apply_texture(heart, HEART_TEX)
    else:
        print("[WARN] Heart texture not found — using red fallback")
        heart.paint_uniform_color([0.85, 0.1, 0.1])
except Exception as e:
    print(f"[WARN] {e} — using fallback sphere for heart")
    heart = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=60)
    heart.paint_uniform_color([0.85, 0.1, 0.1])
heart.translate(-heart.get_center())

# Auto-scale heart to match skull size
skull_extent = np.max(skull.get_axis_aligned_bounding_box().get_extent())
heart_extent = np.max(heart.get_axis_aligned_bounding_box().get_extent())
heart.scale(skull_extent / heart_extent, center=heart.get_center())

# --- OPEN3D WINDOW (3D View — visible to audience) ---
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="HoloGesture — 3D View",
                  width=640, height=640,
                  left=700, top=50,       # position on screen
                  visible=True)
opt = vis.get_render_option()
opt.background_color = np.array([0.05, 0.05, 0.05])  # dark grey bg
opt.light_on = True

current_mesh = skull
vis.add_geometry(current_mesh)
ctr = vis.get_view_control()
ctr.set_zoom(0.8)
last_model = "skull"

# --- OFFSCREEN RENDERER (for 4-view canvas — invisible) ---
vis2 = o3d.visualization.Visualizer()
vis2.create_window(window_name="__hidden__",
                   width=500, height=500,
                   visible=False)
opt2 = vis2.get_render_option()
opt2.background_color = np.array([0, 0, 0])  # pure black for hologram
opt2.light_on = True

current_mesh2 = skull
vis2.add_geometry(current_mesh2)
ctr2 = vis2.get_view_control()
ctr2.set_zoom(0.8)

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

def reset_transform():
    global curr_rot, target_rot, curr_trans, target_trans
    curr_rot     = np.array([0.0, 0.0, 0.0])
    target_rot   = np.array([0.0, 0.0, 0.0])
    curr_trans   = np.array([0.0, 0.0, 0.0])
    target_trans = np.array([0.0, 0.0, 0.0])

# --- 4-VIEW CANVAS BUILDER ---
def make_4view_canvas(frame_bgr):
    h, w = frame_bgr.shape[:2]
    S = min(h, w)
    f = frame_bgr[:S, :S]

    # --- SMART CROP: find object, crop square, center it ---
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, cw, ch = cv2.boundingRect(coords)
        pad = 30
        # Expand to square from center of bounding box
        cx = x + cw // 2
        cy = y + ch // 2
        half = (max(cw, ch) // 2) + pad
        # Clamp to image bounds
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(S, cx + half)
        y2 = min(S, cy + half)
        cropped = f[y1:y2, x1:x2]
        # Force to perfect square
        sq = min(cropped.shape[0], cropped.shape[1])
        cropped = cropped[:sq, :sq]
        f = cropped

    # Fixed panel size — change this for bigger/smaller panels
    PANEL = 300
    f = cv2.resize(f, (PANEL, PANEL))

    front = cv2.flip(f, 0)
    back  = cv2.flip(f, 1)
    left  = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
    right = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Black canvas with centered cross layout
    canvas = np.zeros((PANEL * 3, PANEL * 3, 3), dtype=np.uint8)
    canvas[0:PANEL,         PANEL:PANEL*2]     = front   # top center
    canvas[PANEL:PANEL*2,   0:PANEL]           = left    # middle left
    canvas[PANEL:PANEL*2,   PANEL*2:PANEL*3]   = right   # middle right
    canvas[PANEL*2:PANEL*3, PANEL:PANEL*2]     = back    # bottom center
    return canvas

print("\nControls:")
print("  S key     → Skull")
print("  H key     → Heart")
print("  Move hand → Rotate")
print("  Pinch     → Drag")
print("  2 Hands   → Zoom")
print("  Fist      → Freeze")
print("  L key     → Lock / Unlock")
print("  Q key     → Quit\n")

# --- MAIN LOOP ---
while cap.isOpened():
    with state_lock:
        if should_quit:
            break
        locked = is_locked
        model  = active_model

    # --- MODEL SWITCH ---
    if model != last_model:
        new_mesh = skull if model == "skull" else heart

        # Swap in 3D view
        vis.remove_geometry(current_mesh, reset_bounding_box=False)
        current_mesh = new_mesh
        current_mesh.translate(-current_mesh.get_center())
        vis.add_geometry(current_mesh, reset_bounding_box=True)

        # Swap in hidden renderer
        vis2.remove_geometry(current_mesh2, reset_bounding_box=False)
        current_mesh2 = new_mesh
        vis2.add_geometry(current_mesh2, reset_bounding_box=True)

        reset_transform()
        last_model = model

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

        # FIST = freeze
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
                    current_mesh2.scale(zoom, center=current_mesh2.get_center())
                prev_pinch_dist = curr_dist
            else:
                prev_pinch_dist = None

            prev_hand_pos = idx_tip
    else:
        prev_hand_pos   = None
        prev_pinch_dist = None

    # --- SMOOTH MOVEMENT ---
    # --- SMOOTH MOVEMENT ---
    if not frozen and not locked:
        step_rot = (target_rot - curr_rot) * SMOOTHING_FACTOR
        if np.linalg.norm(step_rot) > 0.001:
            R = current_mesh.get_rotation_matrix_from_xyz(step_rot)
            current_mesh.rotate(R, center=current_mesh.get_center())
            current_mesh2.rotate(R, center=current_mesh2.get_center())
            curr_rot += step_rot

        step_trans = (target_trans - curr_trans) * SMOOTHING_FACTOR
        if np.linalg.norm(step_trans) > 0.001:
            current_mesh.translate(step_trans)
            current_mesh2.translate(step_trans)
            curr_trans += step_trans

    # --- RENDER BOTH (always, even when locked — prevents colour fade) ---
    vis.update_geometry(current_mesh)
    vis.poll_events()
    vis.update_renderer()

    vis2.update_geometry(current_mesh2)
    vis2.poll_events()
    vis2.update_renderer()

    # Capture from hidden renderer for 4-view
    raw      = vis2.capture_screen_float_buffer(do_render=True)
    rendered = (np.asarray(raw)[:, :, ::-1] * 255).astype(np.uint8)

    # --- BUILD 4-VIEW CANVAS ---
    canvas = make_4view_canvas(rendered)

    # Labels on canvas
    model_label = "SKULL" if model == "skull" else "HEART"
    status      = "LOCKED" if locked else ("FROZEN" if frozen else "ACTIVE")
    s_color     = (0,255,80) if locked else ((0,165,255) if frozen else (0,200,255))
    cv2.putText(canvas,
                f"4-VIEW PEPPER'S GHOST  |  {model_label}  |  {status}",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, s_color, 2)
    # Label explaining this window
    cv2.putText(canvas,
                "Place acrylic pyramid on screen below",
                (20, canvas.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Labels on 3D view via driver frame
    cv2.putText(frame, f"Model: {model_label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_color, 2)
    cv2.putText(frame, "S=Skull  H=Heart  L=Lock  Q=Quit",
                (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # --- SHOW ALL 3 WINDOWS ---
    cv2.imshow("HoloGesture — 4-View (Place Pyramid Here)", canvas)
    cv2.imshow("HoloGesture — Driver View", frame)
    vis.poll_events()   # keeps 3D window responsive
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
vis2.destroy_window()
print("[INFO] Session ended.")