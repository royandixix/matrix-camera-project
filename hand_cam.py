import cv2
import mediapipe as mp
import urllib.request
import os
import sys

from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker

MODEL_HAND="hand_landmarker.task"
MODEL_FACE="face_landmarker.task"

if not os.path.exists(MODEL_HAND):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_HAND
    )

if not os.path.exists(MODEL_FACE):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_FACE
    )

FINGER_NAMES={4:"IBU JARI",8:"TELUNJUK",12:"TENGAH",16:"MANIS",20:"KELINGKING"}
FINGER_TIPS=[4,8,12,16,20]
FINGER_MID=[3,6,10,14,18]

CONNECTIONS=[
(0,1),(1,2),(2,3),(3,4),
(0,5),(5,6),(6,7),(7,8),
(5,9),(9,10),(10,11),(11,12),
(9,13),(13,14),(14,15),(15,16),
(13,17),(17,18),(18,19),(19,20),
(0,17)
]

def fingers_up(lm,is_right):
    r=[]
    if is_right:
        if lm[4].x<lm[3].x:r.append(4)
    else:
        if lm[4].x>lm[3].x:r.append(4)
    for t,m in zip(FINGER_TIPS[1:],FINGER_MID[1:]):
        if lm[t].y<lm[m].y:r.append(t)
    return r

def draw_label(img,text,pos,color):
    x,y=pos
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    cv2.rectangle(img,(x-6,y-th-10),(x+tw+6,y+6),(0,0,0),-1)
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

def open_cam():
    for i in range(5):
        cap=cv2.VideoCapture(i,cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            print("Kamera:",i)
            return cap
    return None

hand_options=vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_HAND),
    num_hands=2
)

face_options=vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_FACE),
    num_faces=5
)

hand_detector=HandLandmarker.create_from_options(hand_options)
face_detector=vision.FaceLandmarker.create_from_options(face_options)

cap=open_cam()
if cap is None:
    sys.exit()

cv2.namedWindow("AI Detection",cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Detection",1200,800)

while True:
    ret,frame=cap.read()
    if not ret:
        continue

    frame=cv2.flip(frame,1)
    h,w=frame.shape[:2]

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)

    try:
        hand_result=hand_detector.detect(mp_image)
        face_result=face_detector.detect(mp_image)
    except:
        continue

    # ================= FACE =================
    if face_result.face_landmarks:
        for face in face_result.face_landmarks:
            pts=[(int(l.x*w),int(l.y*h)) for l in face]

            # mata
            for idx in [33,133,362,263]:
                cv2.circle(frame,pts[idx],6,(0,255,255),-1)

            # mulut luar
            for idx in [61,291]:
                cv2.circle(frame,pts[idx],6,(255,0,255),-1)

            # dalam mulut (lidah approx)
            for idx in [13,14]:
                cv2.circle(frame,pts[idx],7,(0,0,255),-1)

            # pipi
            cv2.circle(frame,pts[234],7,(255,150,0),-1)
            cv2.circle(frame,pts[454],7,(255,150,0),-1)

            draw_label(frame,"WAJAH",(pts[10][0],pts[10][1]-10),(0,255,0))

    # ================= HAND =================
    if hand_result.hand_landmarks:
        for i,hand in enumerate(hand_result.hand_landmarks):

            try:
                is_right=hand_result.handedness[i][0].category_name=="Right"
            except:
                is_right=True

            pts=[(int(l.x*w),int(l.y*h)) for l in hand]

            for a,b in CONNECTIONS:
                cv2.line(frame,pts[a],pts[b],(150,150,150),2)

            up=fingers_up(hand,is_right)

            if len(up)==0:
                continue

            for idx,(x,y) in enumerate(pts):
                if idx in FINGER_NAMES:
                    if idx in up:
                        c=(0,255,150)
                        cv2.circle(frame,(x,y),10,c,-1)
                        draw_label(frame,FINGER_NAMES[idx],(x+12,y-12),c)
                    else:
                        cv2.circle(frame,(x,y),4,(100,100,100),-1)
                else:
                    cv2.circle(frame,(x,y),3,(200,200,200),-1)

    cv2.imshow("AI Detection",frame)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()