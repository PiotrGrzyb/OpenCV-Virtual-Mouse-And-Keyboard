import cv2
import mediapipe
import time
import autopy

def main():
    cap = cv2.VideoCapture(0)
    mediapipeHands = mediapipe.solutions.hands
    hands = mediapipeHands.Hands()
    mpDraw = mediapipe.solutions.drawing_utils
    mpDrawStyle = mediapipe.solutions.drawing_styles

    previousTime = 0
    currentTime = 0

    camWidth, camHeight = 940, 880
    cap.set(3, camWidth)
    cap.set(4,camHeight)

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        # If we want to see data that the processors of hands see in terminal
        # print(result.multi_hand_landmarks)
        if result.multi_hand_landmarks:
            for handLandmarks in result.multi_hand_landmarks:
                for id, lm in enumerate(handLandmarks.landmark):
                    #print(id, lm)
                    h , w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    print("[ID:", id , "]","X:", cx,"[Y]", cy)

        if result.multi_hand_landmarks:
            for handLandmarks in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLandmarks, mediapipeHands.HAND_CONNECTIONS)

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Cam", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
