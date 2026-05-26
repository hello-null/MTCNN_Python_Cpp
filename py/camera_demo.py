import cv2
from mtcnn import FaceDetector
from PIL import Image
import numpy as np

detector = FaceDetector()

def camera_detect():
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # 转 PIL
        pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        try:
            # 尝试检测 + 画框
            drawed_pil_im = detector.draw_bboxes(pil_im)
        except ValueError:
            # 没检测到人脸，不画框
            drawed_pil_im = pil_im

        # 转回 OpenCV
        frame = cv2.cvtColor(np.asarray(drawed_pil_im), cv2.COLOR_RGB2BGR)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_detect()