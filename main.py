"""
Drone Flight Simulator (city) with HAND TRACKING controls

SETUP (once):
    py -3.12 -m pip install ursina opencv-python mediapipe==0.10.14

RUN:
    py -3.12 main.py

HAND CONTROLS (webcam, mirrored view, like a selfie):
    LEFT hand  = UP / DOWN   raise hand above the middle = climb
                             lower hand below the middle = descend
                             hand around the middle      = hover
    RIGHT hand = JOYSTICK    move hand up/down    = pitch (forward/back)
                             move hand left/right = roll  (strafe)
                             tilt hand left/right = yaw   (turn)

KEYBOARD (used automatically when a hand is not seen):
    W/S pitch  A/D roll  Q/E turn  SPACE up  SHIFT down  C camera  R reset

Press 'q' while the small camera preview window is selected to hide it.
"""

from ursina import *
from ursina.shaders import lit_with_shadows_shader
import random
import math
import threading
from time import monotonic

import cv2
import mediapipe as mp

# ============================================================
#  SETTINGS  (change these and run again to experiment)
# ============================================================
USE_SHADOWS = True     # set False if shadows look wrong or the game is slow
CITY_SIZE = 4          # city is (2*CITY_SIZE+1) blocks wide
BLOCK = 22             # distance between block centres
NUM_RINGS = 8
MAX_SPEED = 22         # horizontal speed limit
ACCEL = 30             # how hard tilting pushes the drone
CLIMB = 14             # up/down acceleration

# --- hand tracking settings ---
SWAP_HANDS = False     # set True if your left/right hands are swapped
SHOW_PREVIEW = True    # small camera window with the hand skeleton
HAND_TIMEOUT = 0.5     # seconds before a lost hand falls back to keyboard


# ============================================================
#  HAND TRACKING  (runs in a background thread)
# ============================================================
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _axis(v, deadzone=0.12):
    """Clamp to -1..1 and ignore tiny movements near the center."""
    v = _clamp(v, -1.0, 1.0)
    if abs(v) < deadzone:
        return 0.0
    sign = 1.0 if v > 0 else -1.0
    return sign * (abs(v) - deadzone) / (1.0 - deadzone)


class HandControl:
    def __init__(self):
        self.climb = 0.0      # -1..1  (left hand height)
        self.pitch = 0.0      # -1..1  (right hand up/down)
        self.roll = 0.0       # -1..1  (right hand left/right)
        self.yaw = 0.0        # -1..1  (right hand tilt)
        self.show_preview = SHOW_PREVIEW
        self._t_joystick = -999.0
        self._t_climb = -999.0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @property
    def joystick_active(self):
        return monotonic() - self._t_joystick < HAND_TIMEOUT

    @property
    def climb_active(self):
        return monotonic() - self._t_climb < HAND_TIMEOUT

    def _run(self):
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("Hand control: could not open webcam. Using keyboard only.")
            return

        with mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0,  # fastest model, keeps the game smooth
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)  # mirror view
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                h, w, _ = frame.shape

                if results.multi_hand_landmarks:
                    for lms, handed in zip(
                        results.multi_hand_landmarks, results.multi_handedness
                    ):
                        label = handed.classification[0].label  # "Left"/"Right"
                        is_right = (label == "Right") != SWAP_HANDS
                        lm = lms.landmark

                        # Landmark 9 = base of the middle finger (palm center),
                        # landmark 0 = wrist. Coordinates are 0..1 in the frame.
                        cx, cy = lm[9].x, lm[9].y

                        if is_right:
                            # JOYSTICK
                            self.roll = _axis((cx - 0.5) / 0.25)
                            self.pitch = _axis((0.5 - cy) / 0.25)

                            # Hand tilt angle: 0 = fingers straight up
                            dx = lm[9].x - lm[0].x
                            dy = lm[9].y - lm[0].y
                            angle = math.degrees(math.atan2(dx, -dy))
                            self.yaw = _axis(angle / 45.0, deadzone=0.25)

                            self._t_joystick = monotonic()
                            tag = "JOYSTICK"
                        else:
                            # UP/DOWN: hand above the middle climbs, below descends.
                            # Wide deadzone in the middle = easy hover.
                            self.climb = _axis((0.5 - cy) / 0.3, deadzone=0.2)
                            self._t_climb = monotonic()
                            tag = "UP/DOWN"

                        if self.show_preview:
                            mp_draw.draw_landmarks(
                                frame, lms, mp_hands.HAND_CONNECTIONS
                            )
                            cv2.putText(
                                frame, tag,
                                (int(lm[0].x * w) - 40, int(lm[0].y * h) + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                            )

                if self.show_preview:
                    # small center cross = joystick neutral / hover height
                    cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255),
                                   cv2.MARKER_CROSS, 20, 1)
                    cv2.imshow("Hand Control (press q to hide)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cv2.destroyAllWindows()
                        self.show_preview = False  # keep tracking, hide window

        cap.release()


app = Ursina()
window.title = "Drone Flight Simulator"
window.color = color.hsv(210, 0.25, 0.9)
camera.fov = 90

hand = HandControl()

if USE_SHADOWS:
    Entity.default_shader = lit_with_shadows_shader

# ============================================================
#  LIGHTING, SKY, FOG
# ============================================================
Sky()
scene.fog_color = color.hsv(210, 0.25, 0.9)
scene.fog_density = (40, 200)          # (start distance, full-fog distance)

sun = DirectionalLight(shadows=USE_SHADOWS, shadow_map_resolution=(2048, 2048))
sun.look_at(Vec3(1, -1.5, 1))
AmbientLight(color=color.hsv(220, 0.15, 0.55))

# ============================================================
#  CITY
# ============================================================
random.seed(7)                          # same city every run
city = Entity()
ground = Entity(parent=city, model='plane', scale=500, color=color.hsv(0, 0, 0.18))

buildings = []   # each: (centre_x, centre_z, half_width, half_depth, height)

for gx in range(-CITY_SIZE, CITY_SIZE + 1):
    for gz in range(-CITY_SIZE, CITY_SIZE + 1):
        cx, cz = gx * BLOCK, gz * BLOCK

        # pavement under every block
        Entity(parent=city, model='cube', position=(cx, 0.05, cz),
               scale=(BLOCK - 6, 0.1, BLOCK - 6), color=color.hsv(0, 0, 0.45))

        # the centre block is the landing park: no building
        if gx == 0 and gz == 0:
            continue

        w = random.uniform(8, 12)
        d = random.uniform(8, 12)
        dist = math.hypot(gx, gz)
        h = random.uniform(8, 25) + (CITY_SIZE - dist) * random.uniform(2, 7)
        shade = random.uniform(0.35, 0.8)
        tint = color.hsv(random.choice([200, 30, 0, 210, 40]), random.uniform(0.0, 0.25), shade)

        Entity(parent=city, model='cube', position=(cx, h / 2, cz),
               scale=(w, h, d), color=tint,
               texture='white_cube', texture_scale=(w / 2, h / 2))   # grid looks like windows
        # roof slab
        Entity(parent=city, model='cube', position=(cx, h + 0.15, cz),
               scale=(w + 0.4, 0.3, d + 0.4), color=color.hsv(0, 0, 0.25))

        buildings.append((cx, cz, w / 2, d / 2, h))

# landing pad
Entity(parent=city, model='cube', position=(0, 0.15, 0), scale=(5, 0.1, 5), color=color.lime)
Entity(parent=city, model='cube', position=(0, 0.2, 0), scale=(3.5, 0.1, 3.5), color=color.white)

# ============================================================
#  RINGS  (made from small cubes arranged in a circle)
# ============================================================
RING_RADIUS = 3.2
rings = []   # each: dict with entity, parts, state

def make_ring(position):
    root = Entity(position=position)           # ring faces along the z axis
    parts = []
    for i in range(20):
        a = i / 20 * math.tau
        parts.append(Entity(parent=root, model='cube',
                            position=(math.cos(a) * RING_RADIUS, math.sin(a) * RING_RADIUS, 0),
                            scale=(0.9, 0.9, 0.9), rotation_z=math.degrees(a),
                            color=color.azure))
    return {'root': root, 'parts': parts}

def set_ring_color(ring, c):
    for p in ring['parts']:
        p.color = c

def build_rings():
    for r in rings:
        destroy(r['root'])
    rings.clear()
    random.seed(11)
    street_x = [-2 * BLOCK + 11, -BLOCK + 11, 11, BLOCK + 11, 2 * BLOCK + 11]
    for i in range(NUM_RINGS):
        pos = Vec3(random.choice(street_x), random.uniform(5, 14),
                   random.uniform(-CITY_SIZE * BLOCK + 5, CITY_SIZE * BLOCK - 5))
        rings.append(make_ring(pos))

# ============================================================
#  DRONE
# ============================================================
# 'drone' handles position + heading (yaw). 'body' is the visual child that tilts.
drone = Entity(position=(0, 0.5, 0))
body = Entity(parent=drone)

Entity(parent=body, model='cube', scale=(0.9, 0.25, 0.9), color=color.hsv(0, 0, 0.12))
Entity(parent=body, model='cube', scale=(1.9, 0.07, 0.15), rotation_y=45, color=color.hsv(0, 0, 0.3))
Entity(parent=body, model='cube', scale=(1.9, 0.07, 0.15), rotation_y=-45, color=color.hsv(0, 0, 0.3))
Entity(parent=body, model='sphere', scale=0.25, position=(0, -0.05, 0.45), color=color.red)  # front "eye"

props = []
for x in (-0.65, 0.65):
    for z in (-0.65, 0.65):
        Entity(parent=body, model='cube', scale=(0.18, 0.2, 0.18), position=(x, 0.05, z), color=color.black)
        props.append(Entity(parent=body, model='cube', scale=(0.8, 0.02, 0.08),
                            position=(x, 0.18, z), color=color.light_gray))

# state
velocity = Vec3(0, 0, 0)
pitch = 0.0      # forward/back tilt (degrees)
roll = 0.0       # left/right tilt (degrees)
score = 0
next_ring = 0
elapsed = 0.0
state = 'flying'   # 'flying', 'crashed', 'won'
camera_mode = 0
message_timer = 0.0

# ============================================================
#  HUD
# ============================================================
hud = Text(text="", position=(-0.87, 0.47), scale=1.3, background=True)
big_msg = Text(text="", origin=(0, 0), scale=3, y=0.15, color=color.yellow)
help_text = Text(text="Hands: LEFT = up/down, RIGHT = steer   |   Keys: W/S A/D Q/E SPACE SHIFT  C camera  R reset",
                 origin=(0, 0), y=-0.47, scale=1, color=color.white)

# ============================================================
#  GAME FUNCTIONS
# ============================================================
def reset_game():
    global velocity, pitch, roll, score, next_ring, elapsed, state
    drone.position = Vec3(0, 0.5, 0)
    drone.rotation = Vec3(0, 0, 0)
    body.rotation = Vec3(0, 0, 0)
    velocity = Vec3(0, 0, 0)
    pitch = roll = 0.0
    score = 0
    next_ring = 0
    elapsed = 0.0
    state = 'flying'
    big_msg.text = ""
    build_rings()
    refresh_ring_colors()

def refresh_ring_colors():
    for i, r in enumerate(rings):
        if i < next_ring:
            set_ring_color(r, color.hsv(120, 0.6, 0.5))
        elif i == next_ring:
            set_ring_color(r, color.yellow)
        else:
            set_ring_color(r, color.azure)

def crash(reason):
    global state, velocity
    state = 'crashed'
    velocity = Vec3(0, 0, 0)
    big_msg.text = f"CRASH! {reason}\nPress R to restart"
    big_msg.color = color.red

def update():
    global velocity, pitch, roll, score, next_ring, elapsed, state, camera_mode

    dt = time.dt

    # propellers always spin
    for p in props:
        p.rotation_y += 2000 * dt

    if state == 'flying':
        elapsed += dt

        # ---------------- CONTROLS ----------------
        # Each input is -1..1. If a hand is seen it is used, otherwise the keyboard.
        use_joy = hand.joystick_active
        use_climb = hand.climb_active

        pitch_in = hand.pitch if use_joy else (held_keys['w'] - held_keys['s'])
        roll_in = hand.roll if use_joy else (held_keys['d'] - held_keys['a'])
        yaw_in = hand.yaw if use_joy else (held_keys['e'] - held_keys['q'])
        climb_input = hand.climb if use_climb else (held_keys['space'] - held_keys['left shift'])

        # target tilt angles (this is "angle mode", like a camera drone)
        target_pitch = pitch_in * 25
        target_roll = roll_in * 25
        pitch = lerp(pitch, target_pitch, 6 * dt)
        roll = lerp(roll, target_roll, 6 * dt)

        drone.rotation_y += yaw_in * 90 * dt

        # ---------------- PHYSICS ----------------
        yaw = math.radians(drone.rotation_y)
        forward = Vec3(math.sin(yaw), 0, math.cos(yaw))
        right = Vec3(math.cos(yaw), 0, -math.sin(yaw))

        # tilting pushes the drone sideways
        push = (forward * pitch + right * roll) * (ACCEL / 25)
        velocity.x += push.x * dt
        velocity.z += push.z * dt

        # vertical: climb input goes up/down, otherwise the drone hovers
        velocity.y += climb_input * CLIMB * dt
        if climb_input == 0:
            velocity.y *= (1 - 4 * dt)        # hover assist: slow the vertical movement

        # air drag, then speed limit
        velocity.x *= (1 - 1.2 * dt)
        velocity.z *= (1 - 1.2 * dt)
        horizontal = Vec3(velocity.x, 0, velocity.z)
        if horizontal.length() > MAX_SPEED:
            horizontal = horizontal.normalized() * MAX_SPEED
            velocity.x, velocity.z = horizontal.x, horizontal.z
        velocity.y = clamp(velocity.y, -12, 12)

        drone.position += velocity * dt

        # visual tilt (flip the minus sign on roll if it leans the wrong way)
        body.rotation_x = pitch
        body.rotation_z = -roll

        # ---------------- COLLISIONS ----------------
        # ground
        if drone.y < 0.5:
            if velocity.y < -6:
                crash("Landed too hard")
            drone.y = 0.5
            velocity.y = max(velocity.y, 0)

        # ceiling
        drone.y = min(drone.y, 80)

        # buildings: simple box tests (is the drone inside this box?)
        for (cx, cz, hw, hd, h) in buildings:
            if (abs(drone.x - cx) < hw + 0.6 and abs(drone.z - cz) < hd + 0.6 and drone.y < h + 0.6):
                crash("Hit a building")
                break

        # ---------------- RINGS ----------------
        if next_ring < len(rings):
            r = rings[next_ring]['root']
            d = drone.position - r.position
            along = d.z                                   # distance through the ring
            sideways = math.hypot(d.x, d.y)               # distance from the ring's centre
            if abs(along) < 1.2 and sideways < RING_RADIUS - 0.4:
                score += 10
                next_ring += 1
                refresh_ring_colors()
                if next_ring >= len(rings):
                    state = 'won'
                    big_msg.text = f"YOU WIN!  Time: {elapsed:.1f}s\nPress R to play again"
                    big_msg.color = color.lime

    # ---------------- CAMERA ----------------
    yaw = math.radians(drone.rotation_y)
    forward = Vec3(math.sin(yaw), 0, math.cos(yaw))
    if camera_mode == 0:       # chase
        target = drone.position - forward * 7 + Vec3(0, 3, 0)
        camera.position = lerp(camera.position, target, 5 * dt)
        camera.look_at(drone.position + Vec3(0, 0.5, 0))
    elif camera_mode == 1:     # first person
        camera.position = drone.position + Vec3(0, 0.3, 0) + forward * 0.5
        camera.rotation = Vec3(0, drone.rotation_y, 0)   # level horizon: follows turning only, never tilts
    else:                      # top-down
        camera.position = drone.position + Vec3(0, 40, 0)
        camera.rotation = Vec3(90, drone.rotation_y, 0)

    # light follows the drone so shadows stay sharp around it
    sun.position = drone.position + Vec3(-30, 60, -30)

    # ---------------- HUD ----------------
    speed = Vec3(velocity.x, 0, velocity.z).length()
    up_src = "hand" if hand.climb_active else "keys"
    steer_src = "hand" if hand.joystick_active else "keys"
    hud.text = (f"Altitude: {drone.y:.1f} m\nSpeed: {speed:.1f} m/s\n"
                f"Rings: {min(next_ring, len(rings))}/{len(rings)}\n"
                f"Score: {score}\nTime: {elapsed:.1f}s\n"
                f"Up/Down: {up_src}   Steer: {steer_src}")

def input(key):
    global camera_mode
    if key == 'r':
        reset_game()
    if key == 'c':
        camera_mode = (camera_mode + 1) % 3
    if key == 'escape':
        application.quit()

reset_game()
app.run()

