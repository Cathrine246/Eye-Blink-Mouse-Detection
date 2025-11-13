import cv2
import mediapipe as mp
import pyautogui
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

BLINK_RATIO_THRESHOLD = 5.0
CLICK_DELAY = 1.0
last_click_time = 0


screen_w, screen_h = pyautogui.size()


LEFT_EYE_IDS = [33, 160, 158, 133, 153, 144]

NOSE_ID = 1

prev_x, prev_y = 0, 0
SMOOTHING = 0.7 


def get_blink_ratio(landmarks, eye_ids):
    eye = [landmarks[i] for i in eye_ids]
    hor_length = ((eye[0].x - eye[3].x) ** 2 + (eye[0].y - eye[3].y) ** 2) ** 0.5
    ver_length = ((eye[1].x - eye[5].x) ** 2 + (eye[1].y - eye[5].y) ** 2) ** 0.5
    if ver_length == 0:
        return 0
    return hor_length / ver_length


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("📷 Eye Blink Mouse Control started. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

    
        nose = landmarks[NOSE_ID]
        x = int(nose.x * screen_w)
        y = int(nose.y * screen_h)

        smooth_x = int(prev_x * SMOOTHING + x * (1 - SMOOTHING))
        smooth_y = int(prev_y * SMOOTHING + y * (1 - SMOOTHING))
        pyautogui.moveTo(smooth_x, smooth_y)
        prev_x, prev_y = smooth_x, smooth_y

        blink_ratio = get_blink_ratio(landmarks, LEFT_EYE_IDS)
        current_time = time.time()
        clicked = False

        if blink_ratio > BLINK_RATIO_THRESHOLD:
            if current_time - last_click_time > CLICK_DELAY:
                pyautogui.click()
                last_click_time = current_time
                clicked = True

        cv2.putText(frame, f"Blink Ratio: {blink_ratio:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if clicked:
            cv2.putText(frame, "CLICK!", (250, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Eye Blink Mouse Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
