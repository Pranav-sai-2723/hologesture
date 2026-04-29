import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np
import threading
from pynput import keyboard

# --- CONFIGURATION ---
MODEL_PATH = "craneo.obj"
SMOOTHING_FACTOR = 0.20

# --- STATE ---
is_locked = False
should_quit = False
is_frozen = False  # fist-freeze state
state_lock = threading.Lock()

def on_press(key):
    global is_locked, should_quit
    try:
        with state_lock:
            if key.char == 'q': should_quit = True
            elif key.char == 'l': is_locked = not is_locked
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

# --- OPEN3D SETUP ---
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="HoloGesture", width=420, height=300, visible=False)

try:
    skull = o3d.io.read_triangle_mesh(MODEL_PATH)
    skull.compute_vertex_normals()
    if not skull.has_vertex_colors():
        skull.paint_uniform_color([0.9, 0.85, 0.7])
except:
    print("[WARN] Model not found. Using fallback sphere.")
    skull = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=60)
    skull.paint_uniform_color([0.75, 0.75, 0.75])

skull.translate(-skull.get_center())
vis.add_geometry(skull)

opt = vis.get_render_option()
opt.background_color = np.array([0, 0, 0])  # black background
opt.light_on = True

# --- HAND TRACKING ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
cap = cv2.VideoCapture(0)

# --- SMOOTH MOVEMENT VARIABLES ---
curr_rot   = np.array([0.0, 0.0, 0.0])
target_rot = np.array([0.0, 0.0, 0.0])
curr_trans   = np.array([0.0, 0.0, 0.0])
target_trans = np.array([0.0, 0.0, 0.0])

prev_hand_pos   = None
prev_pinch_dist = None

# --- GESTURE HELPERS ---
def is_fist(hand_landmarks):
    """All 4 fingertips below their knuckle = fist = freeze."""
    tips = [8, 12, 16, 20]
    mcps = [5,  9, 13, 17]
    return all(
        hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y
        for tip, mcp in zip(tips, mcps)
    )

def is_pinching(hand_landmarks):
    idx_tip   = np.array([hand_landmarks.landmark[8].x,
                           hand_landmarks.landmark[8].y])
    thumb_tip = np.array([hand_landmarks.landmark[4].x,
                           hand_landmarks.landmark[4].y])
    return np.linalg.norm(idx_tip - thumb_tip) < 0.05

def make_4view_canvas(frame_bgr):
    """
    Take single rendered frame, create 4-view Pepper's Ghost layout.
    All transforms done by OpenCV — no GPU needed.

    Layout on black canvas:
        [    FRONT (flipped vertical)   ]   ← top center
        [ LEFT (rot 90 CW) ][ RIGHT (rot 90 CCW) ]
        [    BACK  (flipped horizontal) ]   ← bottom center
    """
    h, w = frame_bgr.shape[:2]
    S = min(h, w)
    # Crop to square
    f = frame_bgr[:S, :S]

    # 4 transforms
    front = cv2.flip(f, 0)                          # flip vertical
    back  = cv2.flip(f, 1)                          # flip horizontal
    left  = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
    right = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Canvas is 3S wide x 3S tall, all black
    canvas = np.zeros((S * 3, S * 3, 3), dtype=np.uint8)

    # Place views
    canvas[0:S,       S:S*2]   = front   # top center
    canvas[S:S*2,     0:S]     = left    # middle left
    canvas[S:S*2,     S*2:S*3] = right   # middle right
    canvas[S*2:S*3,   S:S*2]   = back    # bottom center

    return canvas

print("Controls:")
print("  Move hand    → Rotate")
print("  Pinch        → Drag")
print("  Two hands    → Zoom")
print("  Fist         → Freeze / Unfreeze")
print("  L key        → Lock / Unlock")
print("  Q key        → Quit")

# --- MAIN LOOP ---
while cap.isOpened():
    with state_lock:
        if should_quit:
            break
        locked = is_locked

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    frozen = False  # reset each frame

    if results.multi_hand_landmarks and not locked:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        hand = results.multi_hand_landmarks[0]

        # --- FIST CHECK (freeze) ---
        if is_fist(hand):
            frozen = True
            prev_hand_pos = np.array([hand.landmark[8].x,
                                      hand.landmark[8].y])
            # Show freeze indicator
            cv2.putText(frame, "FROZEN", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        else:
            idx_tip   = np.array([hand.landmark[8].x, hand.landmark[8].y])
            thumb_tip = np.array([hand.landmark[4].x, hand.landmark[4].y])
            pinching  = is_pinching(hand)

            if prev_hand_pos is not None:
                delta = idx_tip - prev_hand_pos
                if pinching:
                    target_trans[0] += delta[0] * 5
                    target_trans[1] -= delta[1] * 5
                else:
                    target_rot[0] += delta[1] * 10
                    target_rot[1] += delta[0] * 10

            # --- TWO HAND ZOOM ---
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
        prev_hand_pos   = None
        prev_pinch_dist = None

    # --- SMOOTH MOVEMENT (skip if frozen or locked) ---
    if not frozen and not locked:
        step_rot = (target_rot - curr_rot) * SMOOTHING_FACTOR
        if np.linalg.norm(step_rot) > 0.001:
            R = skull.get_rotation_matrix_from_xyz(step_rot)
            skull.rotate(R, center=skull.get_center())
            curr_rot += step_rot

        step_trans = (target_trans - curr_trans) * SMOOTHING_FACTOR
        if np.linalg.norm(step_trans) > 0.001:
            skull.translate(step_trans)
            curr_trans += step_trans

    # --- RENDER OPEN3D ---
    vis.update_geometry(skull)
    vis.poll_events()
    vis.update_renderer()

    # Capture rendered frame from Open3D
    raw = vis.capture_screen_float_buffer(do_render=True)
    rendered = (np.asarray(raw)[:, :, ::-1] * 255).astype(np.uint8)  # RGB→BGR

    # --- BUILD 4-VIEW CANVAS ---
    canvas = make_4view_canvas(rendered)

    # Status overlay on canvas
    status = "LOCKED" if locked else ("FROZEN" if frozen else "ACTIVE")
    color  = (0, 255, 80) if locked else ((0, 165, 255) if frozen else (0, 80, 255))
    cv2.putText(canvas, f"HoloGesture  |  {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Driver view overlay
    cv2.putText(frame, "DRIVER VIEW", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    status_text = "LOCKED" if locked else "ACTIVE"
    cv2.putText(frame, status_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if locked else (0, 0, 255), 2)

    cv2.imshow("HoloGesture — 4-View Pepper's Ghost", canvas)
    cv2.imshow("Driver View", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
print("[INFO] Session ended.")