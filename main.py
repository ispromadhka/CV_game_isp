import cv2,random,time
import mediapipe as mp
from scipy.spatial import distance as dist


mp_hand = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hand.Hands(max_num_hands=6)


ball_radius,gravity,num_balls = 27 , 0.9 , 25


def get_finger_positions(hand_landmarks):
    return {
        "wrist": hand_landmarks.landmark[mp_hand.HandLandmark.WRIST],
        "thumb_tip": hand_landmarks.landmark[mp_hand.HandLandmark.THUMB_TIP],
        "index_tip": hand_landmarks.landmark[mp_hand.HandLandmark.INDEX_FINGER_TIP],
        "middle_tip": hand_landmarks.landmark[mp_hand.HandLandmark.MIDDLE_FINGER_TIP],
        "ring_tip": hand_landmarks.landmark[mp_hand.HandLandmark.RING_FINGER_TIP],
        "pinky_tip": hand_landmarks.landmark[mp_hand.HandLandmark.PINKY_TIP]
    }


def distance(point1, point2):
    return dist.euclidean([point1.x, point1.y], [point2.x, point2.y])

def detect_gesture(finger_positions):
    index_tip = finger_positions["index_tip"]
    wrist = finger_positions["wrist"]
    pinky_tip = finger_positions["pinky_tip"]
    ring_finger_tip = finger_positions["ring_tip"]
    middle_finger_tip = finger_positions["middle_tip"]

    if (distance(index_tip, wrist) < 0.2 and 
        distance(pinky_tip, wrist) < 0.2 and 
        distance(ring_finger_tip, wrist) < 0.2 and 
        distance(middle_finger_tip, wrist) < 0.2):
        return True 
    return False  

def detect_rock_gesture(finger_positions):
    index_tip = finger_positions["index_tip"]
    pinky_tip = finger_positions["pinky_tip"]
    middle_tip = finger_positions["middle_tip"]
    ring_tip = finger_positions["ring_tip"]
    thumb_tip = finger_positions["thumb_tip"]

    if (index_tip.y < middle_tip.y and
        index_tip.y < ring_tip.y and
        pinky_tip.y < middle_tip.y and
        thumb_tip.y > middle_tip.y and
        thumb_tip.y > ring_tip.y):
        return True 
    return False

def detect_pointing_gesture(finger_positions):
    index_tip = finger_positions["index_tip"]
    middle_tip = finger_positions["middle_tip"]
    ring_tip = finger_positions["ring_tip"]
    pinky_tip = finger_positions["pinky_tip"]
    thumb_tip = finger_positions["thumb_tip"]
    if (index_tip.y < middle_tip.y and
        index_tip.y < ring_tip.y and
        index_tip.y < pinky_tip.y and
        index_tip.y < thumb_tip.y):
        return True 
    return False

def get_hand(hand_landmarks, frame_width, frame_height):
    x_min, y_min = pow(10,8), pow(10,8)
    x_max, y_max = -pow(10,8), -pow(10,8)
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * frame_width), int(lm.y * frame_height)
        if x < x_min:
            x_min = x
        if y < y_min:
            y_min = y
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y
    return x_min, y_min, x_max, y_max


balls = [
    {
        'x': random.randint(ball_radius, 640 - ball_radius),
          'y': random.randint(-100, -20),
            'dy': random.uniform(2, 5)
          }
            for _ in range(num_balls)
          ]

cam = cv2.VideoCapture(0)

prev_time,fps,score = 0,0,0
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    if not ret:
        break


    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f'FPS: {int(fps)}',(480,480),cv2.QT_FONT_NORMAL,1, (255,255,255), 1)

    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    for ball in balls:
        ball['y'] += ball['dy']
        ball['dy'] += gravity * 0.1
        if ball['y'] - ball_radius > frame.shape[0]:
            ball['y'] = random.randint(-100, -20)
            ball['dy'] = random.uniform(2, 5)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hand.HAND_CONNECTIONS)
            x_min, y_min, x_max, y_max = get_hand(hand_landmarks, frame_width, frame_height)

            if detect_rock_gesture(get_finger_positions(hand_landmarks)):
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 3)

            if detect_pointing_gesture(get_finger_positions(hand_landmarks)):
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

            if detect_gesture(get_finger_positions(hand_landmarks)):
                balls_to_remove = []
                for i, ball in enumerate(balls):
                    if x_min < ball['x'] < x_max and y_min < ball['y'] < y_max:
                        balls_to_remove.append(i)

                for i in sorted(balls_to_remove, reverse=True):
                    balls.pop(i)
                    score +=1
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)




    for ball in balls:
        cv2.circle(frame, (int(ball['x']), int(ball['y'])), ball_radius, (0, 0, 255), -1)


    cv2.putText(frame, f'Score: {score}', (10, 50), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 2)

    cv2.imshow('Falling Balls', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()