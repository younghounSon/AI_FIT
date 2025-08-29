import os
import cv2
from pathlib import Path

IMG_W, IMG_H = 1920, 1080  # 원본 해상도
# 키포인트 순서 (당신 데이터 기준 24개)
ORDER = [
    "Nose","Left Eye","Right Eye","Left Ear","Right Ear",
    "Left Shoulder","Right Shoulder","Left Elbow","Right Elbow",
    "Left Wrist","Right Wrist","Left Hip","Right Hip",
    "Left Knee","Right Knee","Left Ankle","Right Ankle",
    "Neck","Left Palm","Right Palm","Back","Waist",
    "Left Foot","Right Foot"
]

# 보기 좋은 스켈레톤 연결 (필요에 맞게 추가/수정 가능)
SKELETON = [
    (0,17),                   # Nose - Neck
    (17,5),(17,6),            # Neck - Shoulders
    (5,7),(7,9),              # Left Shoulder - Elbow - Wrist
    (6,8),(8,10),             # Right Shoulder - Elbow - Wrist
    (9,18),(10,19),           # Wrist - Palm
    (17,20),(20,21),(21,11),(21,12),   # Neck-Back-Waist-Hips
    (11,13),(13,15),(15,22),  # Left Hip - Knee - Ankle - Foot
    (12,14),(14,16),(16,23)   # Right Hip - Knee - Ankle - Foot
]

# v 값별 색상 (B,G,R)
COLOR_KPT = {
    2: (0, 255, 0),   # visible: green
    1: (0, 165, 255), # occluded: orange
    0: (200, 200, 200) # none: light gray
}
COLOR_BBOX = (255, 0, 255)  # magenta
COLOR_LINE = (255, 255, 0)  # cyan

def load_yolo_pose_labels(label_path, num_kpts=24):
    """
    한 이미지 라벨(.txt)에서 여러 사람의 줄을 파싱해서
    [{'cls':0,'bbox':(xc,yc,bw,bh),'kpts':[(x,y,v), ...]}] 리스트로 리턴
    """
    persons = []
    if not os.path.exists(label_path):
        return persons
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            vals = line.strip().split()
            cls_id = int(float(vals[0]))
            xc, yc, bw, bh = map(float, vals[1:5])
            k = vals[5:]
            # k 길이 체크 (num_kpts*3)
            if len(k) != num_kpts * 3:
                print(f"[WARN] {label_path} keypoint length mismatch: got {len(k)}, expected {num_kpts*3}")
                continue
            kpts = []
            for i in range(num_kpts):
                x = float(k[3*i+0]) * IMG_W
                y = float(k[3*i+1]) * IMG_H
                v = int(float(k[3*i+2]))
                kpts.append((x, y, v))
            persons.append({"cls": cls_id, "bbox": (xc, yc, bw, bh), "kpts": kpts})
    return persons

def draw_pose(image, persons, draw_names=False):
    """
    이미지 위에 bbox/키포인트/스켈레톤 그리기
    """
    img = image.copy()
    for pid, person in enumerate(persons):
        xc, yc, bw, bh = person["bbox"]
        # 정규화 bbox -> 픽셀 bbox
        xw = bw * IMG_W
        yh = bh * IMG_H
        cx = xc * IMG_W
        cy = yc * IMG_H
        x1 = int(round(cx - xw/2))
        y1 = int(round(cy - yh/2))
        x2 = int(round(cx + xw/2))
        y2 = int(round(cy + yh/2))

        # bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BBOX, 2)
        cv2.putText(img, f"id:{pid}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BBOX, 2, cv2.LINE_AA)

        # skeleton lines (둘 다 v>0 일 때만)
        for a, b in SKELETON:
            xa, ya, va = person["kpts"][a]
            xb, yb, vb = person["kpts"][b]
            if va > 0 and vb > 0:
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), COLOR_LINE, 2, cv2.LINE_AA)

        # keypoints
        for i, (x, y, v) in enumerate(person["kpts"]):
            color = COLOR_KPT.get(v, (255, 255, 255))
            if v > 0:  # v==0이면 안 그림
                cv2.circle(img, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
                if draw_names:
                    name = ORDER[i] if i < len(ORDER) else str(i)
                    cv2.putText(img, name, (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img

def visualize_one(image_path, label_path, out_path=None, show=False):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERR] cannot read: {image_path}")
        return
    persons = load_yolo_pose_labels(label_path, num_kpts=24)
    vis = draw_pose(img, persons, draw_names=False)
    if out_path:
        Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, vis)
    if show:
        cv2.imshow("vis", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

visualize_one("dataset/images/001-1-1-01-Z21_A-0000001.jpg", "dataset/labels/001-1-1-01-Z21_A-0000001.txt",show=True)

# 2) 폴더 일괄 확인
def visualize_folder(img_dir="dataset/images", lbl_dir="dataset/labels", out_dir="debug_vis"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith((".jpg",".jpeg",".png")):
            continue
        label_name = os.path.splitext(img_name)[0] + ".txt"
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(lbl_dir, label_name)
        out_path = os.path.join(out_dir, img_name)
        visualize_one(img_path, label_path, out_path=out_path, show=False)
        print(f"[OK] wrote {out_path}")

# visualize_folder()  # 주석 해제해서 일괄 시각화 생성
