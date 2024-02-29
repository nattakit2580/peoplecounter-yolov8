import math

class Tracker:
    def __init__(self):
        # เก็บตำแหน่งศูนย์กลางของวัตถุ
        self.center_points = {}
        # นับจำนวน ID
        # ทุกครั้งที่ตรวจจับวัตถุ ID ใหม่, จำนวนนี้จะเพิ่มขึ้น 1
        self.id_count = 0

    def update(self, objects_rect):
        # พื้นที่กรอบและ IDs
        objects_bbs_ids = []

        # หาตำแหน่งศูนย์กลางของวัตถุใหม่
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # ตรวจสอบว่าวัตถุนั้นๆได้ถูกตรวจจับแล้วหรือไม่
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    print(f"Detected object with ID: {id}")  # เพิ่มบรรทัดนี้เพื่อพิมพ์ ID
                    break

            # ถ้าวัตถุนั้นๆยังไม่ได้ถูกตรวจจับ, กำหนด ID ใหม่ให้วัตถุนั้น
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # ทำความสะอาดพจนานุกรมตามตำแหน่งศูนย์กลางเพื่อลบ IDS ที่ไม่ได้ใช้
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # อัปเดตพจนานุกรมด้วย IDS ที่ไม่ได้ใช้ถูกลบ
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
