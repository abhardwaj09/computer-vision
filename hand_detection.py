import cv2
import mediapipe as mp
import time
 
class HandDetector:
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands  
        self.detectionCon = detectionCon 
        self.trackCon = trackCon
 
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mp_draw = mp.solutions.drawing_utils
 
    def findHands(self, img, draw = True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
 
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img
 
    def findPosition(self, img, handNumber = 0, draw = True):
 
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(my_hand.landmark):
                height, width, channel = img.shape
                center_x, center_y = int(lm.x * width), int(lm.y * height)
                lm_list.append([id, center_x, center_y])
                if draw:
                    cv2.circle(img, (center_x, center_y), 7, (255, 0, 255), cv2.FILLED)
        return lm_list 
 
def main():
    prev_time, cur_time = 0, 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        _, img = cap.read()
        img = detector.findHands(img)
        lm_list = detector.findPosition(img)
        if len(lm_list) != 0:
            print(lm_list[4])
        cur_time = time.time()
        fps = 1/(cur_time - prev_time)
        prev_time = cur_time
 
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255,), 3) 
        cv2.imshow("Image", img)
        cv2.waitKey(1)
 
 
if __name__ == "__main__":
    main()
 
