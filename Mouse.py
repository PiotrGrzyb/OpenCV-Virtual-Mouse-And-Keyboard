import cv2
import mediapipe


def main():
    cap = cv2.VideoCapture(0)
    mediapipeHands = mediapipe.solutions.hands
    hands = mediapipeHands.Hands()

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        #If we want to see data that the processors of hands see in terminal
        #print(result.multi_hand_landmarks)

        cv2.imshow("Cam", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
