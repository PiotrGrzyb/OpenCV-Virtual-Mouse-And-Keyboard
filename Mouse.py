import cv2
import mediapipe


def main():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        cv2.imshow("Cam", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
