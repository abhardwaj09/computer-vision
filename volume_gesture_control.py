import cv2
import time
import numpy as np
import handDetection as hd
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

camW, camH = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, camW)
cap.set(4, camH)
prevTime = 0

detector = hd.HandDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]
vol, volBar = 0, 400
while True:
    _, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # lengthList = []

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot = x2 - x1, y2 - y1


        vol = np.interp(length, [15, 125], [minVol, maxVol])
        volBar = np.interp(length, [15, 125], [400, 150])
        volume.SetMasterVolumeLevel(vol, None)
        if length < 50:
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.putText(img, "0%", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 255), cv2.FILLED)
    cv2.putText(img, "100%", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime
    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.imshow("img", img)
    cv2.waitKey(1)
