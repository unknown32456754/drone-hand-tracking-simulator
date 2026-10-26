# Drone Flight Simulator with Hand Tracking

A 3D city drone simulator built with [Ursina](https://www.ursinaengine.org/), controlled with your **webcam and hands** using [MediaPipe](https://developers.google.com/mediapipe) hand tracking. Keyboard controls work too and take over automatically when a hand is not visible.

Fly through the rings in order, avoid the buildings, and land gently.

## How it works

- OpenCV reads frames from the webcam.
- MediaPipe's pretrained neural network finds each hand and predicts 21 landmark points.
- Plain code turns those points into flight inputs (pitch, roll, yaw, climb), which the game reads every frame.
- Hand tracking runs in a background thread so the game stays smooth.

## Requirements

- Windows, macOS or Linux with a webcam
- **Python 3.9 to 3.12** (MediaPipe does not support newer versions such as 3.14)

## Install

```bash
pip install -r requirements.txt
```

On Windows, if you have several Python versions installed:

```bash
py -3.12 -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Windows with several Python versions:

```bash
py -3.12 main.py
```

A small camera preview window also opens, showing your hand skeleton. Press `q` in it to hide the preview (tracking keeps running).

## Controls

### Hands (mirrored view, like a selfie)

| Control | What to do | Result |
|---|---|---|
| Left hand up | Raise above the middle of the frame | Climb |
| Left hand down | Lower below the middle | Descend |
| Left hand middle | Keep around the center | Hover |
| Right hand up / down | Move up or down | Pitch forward / back |
| Right hand left / right | Slide sideways | Roll (strafe) |
| Right hand tilt | Rotate like a steering wheel | Yaw (turn) |

### Keyboard

| Key | Action |
|---|---|
| W / S | Pitch forward / back |
| A / D | Roll left / right |
| Q / E | Turn left / right |
| Space | Climb |
| Left Shift | Descend |
| C | Switch camera (chase, first person, top-down) |
| R | Restart |
| Esc | Quit |

## Rules

- Fly through the yellow ring first, then the rest in order. Pass all rings to win.
- Hitting a building is a crash.
- Landing faster than 6 m/s is a crash.

## Settings

Edit the constants at the top of `main.py`:

| Setting | Meaning |
|---|---|
| `SWAP_HANDS` | Set `True` if your left and right hands are swapped |
| `SHOW_PREVIEW` | Show or hide the camera preview window |
| `USE_SHADOWS` | Set `False` if the game is slow |
| `NUM_RINGS`, `MAX_SPEED`, `ACCEL`, `CLIMB` | Game difficulty and feel |

## Troubleshooting

- **Gray camera window with a lock icon (Windows):** turn on Settings > Privacy & security > Camera > "Let desktop apps access your camera".
- **`pip` or `python` not found:** install Python 3.12 from python.org and tick "Add python.exe to PATH".
- **"Application Control policy has blocked this file" (Windows):** Windows Smart App Control or a school/work policy is blocking a library file. Check Windows Security > App & browser control > Smart App Control, or ask your administrator.
- **MediaPipe will not install:** check `python --version`. It must be 3.12 or lower.
- **Controls feel backwards:** open an issue or flip the sign of that input in `update()`.

## Credits

Built with Ursina, OpenCV and MediaPipe.

