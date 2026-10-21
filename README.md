# face_detection

Fun play with face detection and recognition in Python: real-time webcam scripts using OpenCV, dlib and the `face_recognition` library.

## Scripts

| File | What it does |
|------|--------------|
| `face_detect.py` | Opens the default webcam, detects faces in each frame with dlib's frontal face detector (HOG) and draws green boxes around them. |
| `recognition.py` | Loads reference photos from `faces/<name>.jpg` (the list of names is hard-coded as `known_names`), encodes them with `face_recognition`, then labels each face seen by the webcam with the matching name or `Unknown`. |
| `cascade.xml` | OpenCV Haar cascade for frontal faces (Rainer Lienhart's stump-based detector). It is included but not used by the scripts. |

Press `q` in the video window to quit.

## Requirements

```bash
pip install opencv-python numpy dlib face_recognition
```

`dlib` (also required by `face_recognition`) compiles from source on most systems, so CMake and a C++ compiler are needed. The scripts set `QT_QPA_PLATFORM=xcb` for the OpenCV window, which targets Linux/X11 (or XWayland).

## Usage

```bash
python3 face_detect.py

# recognition: create faces/fabio.jpg (or edit known_names) first
mkdir -p faces && cp /path/to/photo.jpg faces/fabio.jpg
python3 recognition.py
```

## License

GNU GPL v3, see [LICENSE](LICENSE).
