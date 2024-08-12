import cv2
import mediapipe as mp
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 4:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                if id == 8:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                if id == 12:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                if id == 16:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
                if id == 20:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            hand = []
            for lm in handLms.landmark:
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand.append((cx, cy))

            if hand[4][1] < hand[3][1]:
                fingerup = 1
            else:
                fingerup = 0

            if hand[8][1] < hand[7][1]:
                fingerup1 = 1
            else:
                fingerup1 = 0

            if hand[12][1] < hand[11][1]:
                fingerup2 = 1
            else:
                fingerup2 = 0

            if hand[16][1] < hand[15][1]:
                fingerup3 = 1
            else:
                fingerup3 = 0

            if hand[20][1] < hand[19][1]:
                fingerup4 = 1
            else:
                fingerup4 = 0

            fingerup = [fingerup, fingerup1, fingerup2, fingerup3, fingerup4]

            if fingerup == [0, 1, 0, 0, 0]:
                fing = cv2.imread(r"C:\Users\venky\Desktop\opencv\1f.jpeg", cv2.IMREAD_UNCHANGED)
                fing = cv2.resize(fing, (220, 280))
                img[50:330, 20:240] = fing

            elif fingerup == [0, 1, 1, 0, 0]:
                fing = cv2.imread(r"C:\Users\venky\Desktop\opencv\2f.jpeg", cv2.IMREAD_UNCHANGED)
                fing = cv2.resize(fing, (220, 280))
                img[50:330, 20:240] = fing

            elif fingerup == [0, 1, 1, 1, 0]:
                fing = cv2.imread(r"C:\Users\venky\Desktop\opencv\3f.jpeg", cv2.IMREAD_UNCHANGED)
                fing = cv2.resize(fing, (220, 280))
                img[50:330, 20:240] = fing
            elif fingerup == [0, 1, 1, 1, 1]:
                fing = cv2.imread(r"C:\Users\venky\Desktop\opencv\4f.jpeg", cv2.IMREAD_UNCHANGED)
                fing = cv2.resize(fing, (220, 280))
                img[50:330, 20:240] = fing
            elif fingerup == [1, 1, 1, 1, 1]:
                fing = cv2.imread(r"C:\Users\venky\Desktop\opencv\four f and 1 t.jpeg", cv2.IMREAD_UNCHANGED)
                fing = cv2.resize(fing, (220, 280))
                img[50:330, 20:240] = fing

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()