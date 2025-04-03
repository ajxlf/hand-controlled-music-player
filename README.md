# Hand-Controlled Audio Player

An interactive Python-based audio player that uses your hands to control playback speed and volume through your webcam. Featuring real-time gesture tracking, animated visual feedback, and intuitive pinch-based controls, this project combines computer vision and audio processing into a fun and futuristic experience.

---

## 🎯 Features

- **Playback Speed Control**: Adjust speed by changing the distance between your hands.
- **Volume Control**: Pinch with your left hand to control the volume level.
- **Play/Pause Gesture**: Pinch with your right hand to toggle playback.
- **Visual Feedback**: Dynamic wave bars between hands animate with music and respond to hand gestures.
- **Progress Bar**: Displays current playback progress at the bottom of the screen.

---

## 🛠️ Requirements

- Python 3.9+
- OpenCV
- MediaPipe
- NumPy
- sounddevice
- scipy
- pydub

Install dependencies:

```bash
pip install opencv-python mediapipe numpy sounddevice scipy pydub
```

Also install ffmpeg if you're using .mp3:

```bash
# Windows (choco):
choco install ffmpeg

# Mac (brew):
brew install ffmpeg
```

---

## 🚀 How to Run

1. Place your audio file in the project folder and update the `audio_path` in the script.
2. Add `play.png` and `pause.png` icons to the project directory.
3. Run the Python script:

```bash
python hand_controlled_player.py
```

4. Use your webcam and hands to interact:
   - 🖐️ Move hands apart to speed up.
   - 🤏 Left hand pinch = Volume control.
   - 🤏 Right hand pinch = Play/Pause toggle.

---

## 📸 Screenshots

*Add demo screenshots or GIFs here to showcase the wave bars and UI.*

---

## 💡 Ideas for Expansion

- Spotify/YouTube Music integration
- Add gesture-based track skipping
- Save hand movement as visual animations
- Convert to desktop app (PyInstaller) or mobile (Kivy)

---

## 📄 License

MIT License

---

## 🤝 Contributions

Pull requests and suggestions welcome! Open an issue or fork the repo.

---

Built with ❤️ using OpenCV, MediaPipe, and a bit of musical flair.
