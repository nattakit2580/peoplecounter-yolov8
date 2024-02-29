import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

# โมเดล YOLO
model = YOLO('yolov8s.pt')

# กำหนดพื้นที่ที่สนใจ
# ขยับ area1 และ area2 ทางขวามือ (เพิ่มค่าในทิศ x)
shift_value = 280  # ปรับค่าตามที่คุณต้องการ
area1 = [(x + shift_value, y) for x, y in [(172, 348), (149, 350), (404, 369), (427, 362)]]
area2 = [(x + shift_value, y) for x, y in [(139, 352), (110, 357), (353, 377), (384, 369)]]

# ฟังก์ชันตรวจจับเมาส์
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# สร้างหน้าต่าง "RGB" และตั้งค่าการตรวจจับเมาส์
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# เปิดวิดีโอ
cap = cv2.VideoCapture('7-ELEVEN3.mp4')

# อ่านรายชื่อคลาสจากไฟล์ coco.txt
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# ตัวแปรนับเฟรม
count = 0

tracker = Tracker()

people_entering = {}
entering = set()

people_exiting = {}
exiting = set()
# วนลูปการประมวลผลทุกเฟรมในวิดีโอ
while True:
    # อ่านเฟรมจากวิดีโอ
    ret, frame = cap.read()

    # ตรวจสอบว่าอ่านเฟรมสำเร็จหรือไม่
    if not ret:
        break

    # นับเฟรม
    count += 1

    # ข้ามเฟรมที่ไม่จำเป็น (แสดงเฉพาะเฟรมที่มีค่าเฟรมคู่)
    if count % 2 != 0:
        continue

    # ปรับขนาดเฟรม
    frame = cv2.resize(frame, (1020, 500))

    # ทำนายวัตถุในเฟรม
    results = model.predict(frame)

    # ดึงข้อมูลของกล่องที่รอบตัววัตถุ
    a = results[0].boxes.data
    a = a.cpu().numpy()  # ย้ายข้อมูลจาก GPU (CUDA) ไปยัง CPU และแปลงเป็น NumPy array
    px = pd.DataFrame(a).astype("float")
    list = []

    # วนลูปทุกกล่องที่ตรวจจับได้
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        # ถ้าเป็นวัตถุ 'person' ให้วาดสี่เหลี่ยมและแสดงชื่อ
        if 'person' in c:
            list.append([x1, y1, x2, y2])
            # ถ้าวัตถุที่ตรวจจับเป็น 'person' ให้เพิ่มข้อมูลของกล่องนั้นในลิสต์ list.
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        # cv2.pointPolygonTest เพื่อตรวจสอบว่าจุดศูนย์กลางของวัตถุ ((x4, y4)) อยู่ภายในพื้นที่ที่กำหนด (area2)หรือไม่ ถ้าวัตถุอยู่ในพื้นที่จะทำการบันทึกตำแหน่งและแสดงสี่เหลี่ยมสีแดงบนเฟรม.
        results = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
        if results >= 0:
            people_entering[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            
        # ตรวจสอบการเคลื่อนไหวของวัตถุที่มี ID ที่เข้ามา
        if id in people_entering:
            #ใช้ cv2.pointPolygonTest เพื่อตรวจสอบว่าจุดศูนย์กลางของวัตถุ ((x4, y4)) อยู่ภายใน (results >= 0) พื้นที่ที่กำหนด (area1).
            results = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
            if results >= 0:
                #ถ้าวัตถุอยู่ในพื้นที่ที่กำหนด, จะทำการแสดงผลต่างๆบนเฟรม เช่น วาดสี่เหลี่ยมสีเขียว
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                # วาดวงกลมสีชมพู
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                # และแสดงข้อความ ID 
                cv2.putText(frame, str(id), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                # ID ของวัตถุจะถูกเพิ่มลงในเซ็ต entering เพื่อติดตามว่าวัตถุนั้นๆได้เข้ามาในพื้นที่ที่กำหนด.
                entering.add(id)

        ### people_exiting ###
        # ตรวจสอบว่าจุดศูนย์กลางของวัตถุ ((x4, y4)) อยู่นอกพื้นที่ที่กำหนด (area1).
        results2 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
        if results2 >= 0:
            people_exiting[id] = (x4, y4)
            #  ถ้าเป็นเช่นนั้น จะทำการบันทึกตำแหน่งของวัตถุและแสดงสี่เหลี่ยมสีเขียวบนเฟรม.
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if id in people_exiting:
            # ตรวจสอบว่าจุดศูนย์กลางของวัตถุ ((x4, y4)) อยู่ภายใน (results3 >= 0) พื้นที่ที่กำหนด (area2).
            results3 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
            if results3 >= 0:
                # # บันทึกและแสดงผลต่างๆบนเฟรม เช่น วาดสี่เหลี่ยมสีม่วง
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                # วาดวงกลมสีชมพู (cv2.circle), และแสดง ID 
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                # นอกจากนี้, ID ของวัตถุที่ออกจากร้านจะถูกเพิ่มลงในเซ็ต exiting  เพื่อติดตามว่าวัตถุนั้นๆได้ออกจากร้าน.
                cv2.putText(frame, str(id), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                exiting.add(id)

    # วาดหน้าต่างพื้นที่ที่สนใจบนเฟรม
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, '1', (444 + shift_value, 371), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    # วาดหน้าต่างพื้นที่ที่สนใจเพื่อการนับคน
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, '2', (406 + shift_value, 385), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
    # print(people_entering)
    i = (len(entering))
    o = (len(exiting))

    # แสดงจำนวนคนเข้าร้าน
    cv2.putText(frame, f"People entering the store: {str(i)}", (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    # แสดงจำนวนคนที่อยู่ในร้าน
    cv2.putText(frame, f"People leaving the store: {str(o)}", (60, 110), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    # แสดงจำนวนคนที่อยู่ในร้าน
    cv2.putText(frame, f"people in the store: {str(len(tracker.center_points))}", (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    # แสดงเฟรมที่ผ่านการประมวลผล
    cv2.imshow("RGB", frame)

    # หยุดลูปหากผู้ใช้กด 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ปิดหน้าต่าง
cap.release()
cv2.destroyAllWindows()
