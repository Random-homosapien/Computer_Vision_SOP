import cv2
import argparse
from mediapipe.python.solutions import face_detection
import os

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#function
def process(img, face_detection):
    height, width = img.shape[0], img.shape[1]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_out = face_detection.process(img_rgb)
    # print(img_out.detections)
    if img_out.detections is not None:
        for detection in img_out.detections:
            l_data = detection.location_data
            bounding_box = l_data.relative_bounding_box
            x1, y1, w, h = bounding_box.xmin, bounding_box.ymin, bounding_box.width, bounding_box.height
            x1, y1, w, h = int(x1*width), int(y1*height), int(w*width), int(h*height)
            cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0,255,0), 1)

            #blur Faces
            # img[y1:y1+h, x1:x1+w] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (40,40))
            return(img)
    else:
        print("face not found")
    
    
    
    # for detection in img_out.detections:
    #     l_data = detection.location_data
    #     l_data = l_data.relative_keypoints
    #     for j in l_data:
    #         # print(f"{j = }\t{type(j)}\n")
    #         # img_out[j] = [255,0,0]
    #         print(f"{j.x}\n {j.y}\n")
    #         x1,y1 = j.x, j.y
    #         # x1, y1 = int(x1*img.shape[1]), int(y1*img.shape[0])
    #         print(x1, y1)
    #         print(type(x1))
    #         img[x1-5:x1+5 ,y1-5:y1+5] = [255,0,0]
    #     # print(f"{l_data = }")
    # print(type(img_out.detections))

args = argparse.ArgumentParser()
args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)
args = args.parse_args()

#read Images
img_path = "Image Processing/Resources/WIN_20241218_22_05_22_Pro.mp4"
img_path2 = "Image Processing/Resources/WIN_20241223_21_35_48_Pro.jpg"
# img_path3 = "Image Processing\Resources\Lock.jpg"
img = cv2.imread(img_path2)

# print(img.shape[0], img.shape[1])
# cv2.imshow("test", img)
# cv2.waitKey(0)
#detect Faces

mp_face_detection = face_detection

with mp_face_detection.FaceDetection(min_detection_confidence = 0.5, model_selection = 0) as face_detection:
    if args.mode in ['image']:
        img = cv2.imread(args.filePath)

        img = process(img, face_detection)
        cv2.imshow("pts", img)
        cv2.waitKey(0)

        #saving Image
        cv2.imwrite(os.path.join(output_dir, "output.jpg"), img)
    
    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        while ret:
            frame = process(frame, face_detection)

            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('d'):
                break

            ret, frame = cap.read()

        cap.release()