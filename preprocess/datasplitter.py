import os, shutil, random
from pathlib import Path

# 원본 경로
img_dir = Path("dataset/images")
lbl_dir = Path("dataset/labels")

# 새 경로
train_img_dir = Path("dataset/images/train")
val_img_dir   = Path("dataset/images/val")
train_lbl_dir = Path("dataset/labels/train")
val_lbl_dir   = Path("dataset/labels/val")

# 폴더 생성
for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    d.mkdir(parents=True, exist_ok=True)

# 이미지 목록 수집
images = [f for f in img_dir.glob("*.jpg")]  # 필요시 png/jpeg도 추가
random.seed(42)  # 재현성
random.shuffle(images)

# 8:2 split
split_idx = int(len(images) * 0.8)
train_files = images[:split_idx]
val_files   = images[split_idx:]

def move_pair(img_path, dst_img_dir, dst_lbl_dir):
    name = img_path.stem
    lbl_path = lbl_dir / (name + ".txt")
    # 이미지 복사
    shutil.move(str(img_path), str(dst_img_dir / img_path.name))
    # 라벨 복사 (존재한다면)
    if lbl_path.exists():
        shutil.move(str(lbl_path), str(dst_lbl_dir / lbl_path.name))

# 이동 실행
for f in train_files:
    move_pair(f, train_img_dir, train_lbl_dir)

for f in val_files:
    move_pair(f, val_img_dir, val_lbl_dir)

print(f"총 {len(images)} 장 중 {len(train_files)} train, {len(val_files)} val 로 분리 완료!")
