from ultralytics import YOLO
import torch,os

if __name__ == '__main__':
    DATA = "data.yaml"
    model = YOLO("yolo11n-pose.pt")
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    model = YOLO("yolo11s-pose.pt")   # COCO 사전학습 체크포인트
    results1 = model.train(
        data=DATA,
        imgsz=960,                 # 느리면 640/512로
        epochs=15,
        batch=16,                  # GPU 메모리 따라 조절
        device=DEVICE,
        lr0=5e-4, lrf=0.1,         # cosine 스케줄 (기본 on)
        warmup_epochs=3,
        amp=True,                  # fp16
        freeze="backbone",         # 백본 동결
        workers=8,
    )

    ckpt = model.ckpt_path or "runs/pose/train/weights/last.pt"

    # -------- 2단계: 언프리즈(정밀 튜닝) --------
    model2 = YOLO(ckpt)            # 1단계 결과에서 이어서
    results2 = model2.train(
        data=DATA,
        imgsz=960,
        epochs=40,
        batch=16,
        device=DEVICE,
        lr0=3e-4, lrf=0.05,
        warmup_epochs=1,
        amp=True,
        workers=8,
        # freeze=None  # (기본값) 전층 학습
    )

    # -------- 검증(통합 val) --------
    val_res = model2.val(data=DATA, imgsz=960, device=DEVICE)
    print(val_res)  # OKS-AP 등 지표 출력