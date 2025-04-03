import cv2
import mediapipe as mp
import numpy as np
import threading
import sounddevice as sd
from scipy.signal import resample
from pydub import AudioSegment

# Load and normalize audio
audio_path = r"paste path to your audio file" #update this line
song = AudioSegment.from_file(audio_path).set_channels(1).set_frame_rate(44100)
audio_data: np.ndarray = np.array(song.get_array_of_samples()).astype(np.float32)
audio_data /= np.max(np.abs(audio_data))
sample_rate = 44100

# Load play/pause icons
play_icon = cv2.resize(cv2.imread("play.png", cv2.IMREAD_UNCHANGED), (60, 60))
pause_icon = cv2.resize(cv2.imread("pause.png", cv2.IMREAD_UNCHANGED), (60, 60))

# Playback and visual state variables
speed = 1.0
volume = 1.0
paused = True
started = False
pointer = 0
block_size = 1024

# Threading locks
speed_lock = threading.Lock()
pause_lock = threading.Lock()
last_left_pinch = False

# Waveform animation setup
num_bars = 0
wave_heights = []

# Utility functions
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def map_distance_to_speed(distance, min_dist=30, max_dist=150, min_speed=0.5, max_speed=2.0):
    distance = max(min_dist, min(distance, max_dist))
    return ((distance - min_dist) / (max_dist - min_dist)) * (max_speed - min_speed) + min_speed

def overlay_transparent(background, overlay, x, y):
    if overlay.shape[2] < 4:
        return background
    h, w = overlay.shape[:2]
    if x + w > background.shape[1] or y + h > background.shape[0]:
        return background
    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    roi = background[y:y+h, x:x+w]
    blended = roi * (1 - mask) + overlay_img * mask
    background[y:y+h, x:x+w] = blended.astype(np.uint8)
    return background

def get_volume_from_hand(thumb, index):
    dist = calculate_distance(*thumb, *index)
    return np.clip((dist - 20) / 100, 0.1, 1.0)

def draw_wave_bars_between_hands(image, wave_heights, point1, point2, bar_width):
    color = (0, 255, 0) if not paused else (100, 100, 100)
    vector = np.array(point2) - np.array(point1)
    length = np.linalg.norm(vector)
    direction = vector / length if length != 0 else np.array([1.0, 0.0])
    spacing = length / max(len(wave_heights), 1)
    normal = np.array([-direction[1], direction[0]])
    for i, height in enumerate(wave_heights):
        center = np.array(point1) + direction * (i * spacing)
        p1 = center - normal * (height)
        p2 = center + normal * (height)
        cv2.line(image, tuple(p1.astype(int)), tuple(p2.astype(int)), color, bar_width)
    return image

def update_wave_heights(volume):
    for i in range(len(wave_heights)):
        base_height = np.random.randint(10, 40)
        scale = volume ** 1.5
        wave_heights[i] = int(base_height * scale + 5)

def draw_progress_bar(image, pointer, audio_length, sample_rate):
    bar_width = 400
    bar_height = 20
    bar_x = 10
    bar_y = image.shape[0] - 40
    progress = pointer / audio_length
    filled = int(bar_width * progress)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height), (0, 200, 0), -1)
    time_text = f"{int(pointer / sample_rate)}s / {int(audio_length / sample_rate)}s"
    cv2.putText(image, time_text, (bar_x + bar_width + 10, bar_y + bar_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image

def callback(outdata, frames, time_info, status):
    global pointer
    with pause_lock:
        if paused or not started:
            outdata[:] = np.zeros((frames, 1), dtype=np.float32)
            return
    with speed_lock:
        current_speed = speed
    start = int(pointer)
    end = int(start + block_size * current_speed)
    if end > len(audio_data):
        outdata[:] = np.zeros((frames, 1))
        return
    chunk = audio_data[start:end]
    resampled = resample(chunk, frames).astype(np.float32) * volume
    outdata[:, 0] = resampled
    pointer += block_size * current_speed

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Audio and camera
stream = sd.OutputStream(callback=callback, samplerate=sample_rate, channels=1, dtype='float32', blocksize=block_size)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    h, w, _ = image.shape

    if len(wave_heights) == 0:
        num_bars = max(10, w // 30)
        wave_heights = [0] * num_bars

    left_center = right_center = None
    left_thumb = left_index = right_thumb = right_index = None
    hand_types = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, handLms in enumerate(results.multi_hand_landmarks):
            hand_type = results.multi_handedness[i].classification[0].label
            hand_types.append(hand_type)
            thumb = handLms.landmark[4]
            index = handLms.landmark[8]
            thumb_pos = (int(thumb.x * w), int(thumb.y * h))
            index_pos = (int(index.x * w), int(index.y * h))
            distance = calculate_distance(*thumb_pos, *index_pos)

            if hand_type == "Left":
                left_center = ((thumb_pos[0] + index_pos[0]) // 2, (thumb_pos[1] + index_pos[1]) // 2)
                left_thumb, left_index = thumb_pos, index_pos
                volume = get_volume_from_hand(left_thumb, left_index)
                cv2.line(image, left_thumb, left_index, (0, 255, 255), 2)
                cv2.putText(image, f"Volume: {volume:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elif hand_type == "Right":
                right_center = ((thumb_pos[0] + index_pos[0]) // 2, (thumb_pos[1] + index_pos[1]) // 2)
                right_thumb, right_index = thumb_pos, index_pos
                is_pinch = distance < 40
                if is_pinch and not last_left_pinch:
                    if not started:
                        stream.start()
                        started = True
                        paused = False
                    else:
                        with pause_lock:
                            paused = not paused
                last_left_pinch = is_pinch

            mp_draw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)

    if left_center and right_center:
        hand_distance = calculate_distance(*left_center, *right_center)
        new_speed = map_distance_to_speed(hand_distance, min_dist=50, max_dist=300)
        with speed_lock:
            speed = new_speed
        cv2.putText(image, f"Speed: {new_speed:.2f}x", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        image = draw_wave_bars_between_hands(image, wave_heights, left_center, right_center, bar_width=5)

    elif right_center:
        with speed_lock:
            speed = 1.0
        cv2.putText(image, "Right hand not detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    elif left_center:
        volume = get_volume_from_hand(left_thumb, left_index)
        cv2.putText(image, "Left hand not detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"Volume: {volume:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    else:
        with speed_lock:
            speed = 1.0
        volume = 1.0
        cv2.putText(image, "Hands not detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if started and not paused:
        update_wave_heights(volume)

    if started and not paused:
        image = overlay_transparent(image, play_icon, 10, 200)
    if paused:
        image = overlay_transparent(image, pause_icon, 10, 200)

    if started:
        image = draw_progress_bar(image, pointer, len(audio_data), sample_rate)

    cv2.imshow("Hand-Controlled Playback", image)
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

stream.stop()
stream.close()
cap.release()
cv2.destroyAllWindows()
