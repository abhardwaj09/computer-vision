import time
import cv2
import mediapipe as mp


class FaceMeshDetector:

    def __init__(self, static_image_mode=False, maxNumFaces=2, minDetectionCon=0.5, minTrackingCon=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = maxNumFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode, maxNumFaces, minDetectionCon, minTrackingCon)
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            faces = []
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpecs,
                                               self.drawSpecs)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([id, x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    prevTime = 0
    detector = FaceMeshDetector()
    while True:
        _, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))
        curTime = time.time()
        fps = 1 / (prevTime - curTime)
        prevTime = curTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0,), 3)
        cv2.imshow("img", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
