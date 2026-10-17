# Hand-Controlled Audio Player

A webcam-controlled audio player built with Python, OpenCV and MediaPipe. Hand gestures control playback speed, volume and play/pause, with animated visual feedback.

## Features

- **Playback speed:** move your hands closer together or farther apart to adjust speed
- **Volume:** use a pinch gesture with your left hand
- **Play/pause:** use a pinch gesture with your right hand
- **Visual feedback:** animated waveform bars respond to the distance between your hands and the playback state
- **Playback progress:** a progress indicator shows the current position in the track

## Technology

- Python
- OpenCV
- MediaPipe
- NumPy
- sounddevice
- SciPy
- pydub

## Prerequisites

- Python 3.9 or later
- A working webcam
- Audio output
- FFmpeg installed if you intend to load MP3 or other FFmpeg-supported formats through pydub

## Setup and run

1. Clone or download this repository and open a terminal in the project directory.
2. Create and activate a virtual environment (recommended).

   **Windows**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS / Linux**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the Python packages:

   ```bash
   pip install opencv-python mediapipe numpy sounddevice scipy pydub
   ```

4. Open `Music Player.py` and replace the placeholder value of `audio_path` near the top of the file with the path to an audio file on your computer.

   **Windows example**
   ```python
   audio_path = r"C:\Users\YourName\Music\track.mp3"
   ```

   **macOS / Linux example**
   ```python
   audio_path = "/Users/yourname/Music/track.mp3"
   ```

5. Keep `play.png` and `pause.png` in the same directory as `Music Player.py`; these image files are used by the interface.
6. Run the script:

   ```bash
   python "Music Player.py"
   ```

The webcam and audio device must be available to Python. If an MP3 fails to load, confirm that FFmpeg is installed and available on your system path.

## Controls

| Gesture | Action |
|---|---|
| Move both hands farther apart | Increase playback speed |
| Move both hands closer together | Decrease playback speed |
| Pinch with the left hand | Adjust volume |
| Pinch with the right hand | Toggle play/pause |

## Current limitations

The audio file is selected by editing a local path in the source code. A packaged installer, in-app file picker and recorded demo are not included.

## Licence

A separate licence file is not currently included. Confirm the intended licence and add a `LICENSE` file with the correct copyright holder before redistributing this project.
