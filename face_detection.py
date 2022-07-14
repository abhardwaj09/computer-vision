import time
import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(minDetectionCon)

    def detectFace(self, img, draw = True):
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        self.results = self.face_detection.process(img_rgb)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox= int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    self.mp_draw.draw_detection(img, detection)
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, f"{int(detection.score[0]*100)} %", (bbox[0], bbox[1]- 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return img, bboxs 

def main():

    cap = cv2.VideoCapture(0)
    prev_time = 0
    detector = FaceDetector()

    while True:
        _, img = cap.read()
        img, bboxs = detector.detectFace(img)
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        prev_time = cur_time
        cv2.imshow("IMAGE", img)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()