import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np
import threading
from pynput import keyboard

# --- CONFIGURATION ---
MODEL_PATH = "craneo.obj"  
SMOOTHING_FACTOR = 0.20         
ROTATION_SENSITIVITY = 0.009
ZOOM_SENSITIVITY = 0.009

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Skull", width=1280, height=720)

try:
    skull = o3d.io.read_triangle_mesh(MODEL_PATH)
    skull.compute_vertex_normals()
    if not skull.has_vertex_colors():
        skull.paint_uniform_color([0.9, 0.85, 0.7])  # bone colour
except:
    print("[WARN] Skull model not found. Using high-res fallback.")
    skull = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=100)
    skull.paint_uniform_color([0.7, 0.7, 0.7])

skull.translate(-skull.get_center())  # Center it
vis.add_geometry(skull)

# --- HAND TRACKING SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
cap = cv2.VideoCapture(0)

# VARIABLES FOR SMOOTHNESS ---
curr_rot = np.array([0.0, 0.0, 0.0])
target_rot = np.array([0.0, 0.0, 0.0])
curr_trans = np.array([0.0, 0.0, 0.0])
target_trans = np.array([0.0, 0.0, 0.0])

prev_hand_pos = None
prev_pinch_dist = None

print("Controls: \n- Move Hand: Rotate\n- Pinch (Thumb+Index): Drag\n- Two Hand Distance: Zoom\n- 'Q': Quit\n- 'L': Lock/Unlock Controls")

is_locked = False
should_quit = False
state_lock = threading.Lock()

def is_fist(hand_landmarks):
    # Check if all 4 fingertips are below their MCP (knuckle) joints
    tips   = [8, 12, 16, 20]   # index, middle, ring, pinky tips
    mcps   = [5,  9, 13, 17]   # their base knuckles
    return all(
        hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y
        for tip, mcp in zip(tips, mcps)
    )

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
while cap.isOpened():
    with state_lock:
        if should_quit:
            break
        locked = is_locked
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and not locked:

        mp_drawing = mp.solutions.drawing_utils
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        hand = results.multi_hand_landmarks[0]
        idx_tip = np.array([hand.landmark[8].x, hand.landmark[8].y])
        thumb_tip = np.array([hand.landmark[4].x, hand.landmark[4].y])

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

    vis.update_geometry(skull)
    vis.poll_events()
    vis.update_renderer()

    cv2.putText(frame, "OBJECT INTERFACE ACTIVE", (20, 40), 1, 1.5, (0, 0, 255), 2)
    status = "LOCKED" if locked else "UNLOCKED"
    cv2.putText(frame, status, (20, 80), 1, 1.5,
                (0, 255, 0) if is_locked else (0, 0, 255), 2)

    cv2.imshow("Driver View", frame)

    cv2.waitKey(1)  # only keeps OpenCV window alive, no longer handles input

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()