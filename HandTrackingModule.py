import cv2
import time
import mediapipe as mp
from mediapipe.python.solutions.hands import Hands


cap = cv2.VideoCapture(0)



class handDetector():
    def __init__(self,
            mode = False,
            maxHands = 2,
            detectionCon = 0.5,
            trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw= True):
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRgb)
        # debug
        # print(results.multi_hand_landmarks)
        
        hand_landmarks = self.results.multi_hand_landmarks
        if hand_landmarks:
            for handldms in hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handldms,self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo = 0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c  = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 15, (255,0,0), cv2.FILLED)
        
        return lmList


def main():
    pTime = 0
    detector = handDetector()
    while True:
        success, img = cap.read()
        detector.findHands(img)
        lmList = detector.findPosition(img, draw = False)
        if len(lmList) != 0:
            print(lmList)

        cTime = time.time()
        fps = 1/ (cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)


        cv2.imshow("Image", img)
        cv2.waitKey(1) 


if __name__ == "__main__":
    main()