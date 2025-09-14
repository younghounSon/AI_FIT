# main.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import PoseDataset
from arch import TCNModel


def sigmoid_accuracy(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    멀티라벨 cond 용 간단 정확도: per-element 일치 비율
    logits: (N, C), targets: (N, C) with {0,1}
    mask:   (N, C) bool, 유효한 항목만 집계
    """
    preds = (logits.sigmoid() > thr).to(targets.dtype)
    if mask is not None:
        mask = mask.to(torch.bool)
        num = ((preds == targets).float() * mask.float()).sum()
        den = mask.float().sum().clamp_min(1.0)
        return num / den
    return (preds == targets).float().mean()


class BaseModel:
    def __init__(self,
        num_workers: int = 4,
        milestones=(10, 20),
        gamma: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model (arch.TCNModel는 새 프로토타입 버전이어야 함)
        self.model = TCNModel().to(self.device)
        self.model.train()

        # Loss
        self.CELoss = nn.CrossEntropyLoss()        # cls (single-label)

        # Optimizer / Scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=list(milestones), gamma=gamma
        )

        # Dataloader config
        self.num_workers = num_workers
        self.pin_memory = True if self.device == 'cuda' else False

    def make_dataloader(self, data_path: str, mapper: str, batch_size=8, shuffle=True) -> DataLoader:
        dataset = PoseDataset(data_path, mapper)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def _move_to_device(self, *tensors):
        return [t.to(self.device, non_blocking=True) for t in tensors]

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        입력이 (N, T, 24, 2)면 (N, T, 48)로 평탄화.
        이미 (N, T, 48)인 경우 그대로 사용.
        """
        if inputs.dim() == 4 and inputs.size(-1) == 2:
            return inputs.flatten(2, 3)
        return inputs

    @staticmethod
    def _cls_to_index(cls: torch.Tensor) -> torch.Tensor:
        """
        cls가 (N, C) one-hot이면 argmax로 (N,) 인덱스로 변환.
        이미 (N,)이면 그대로 반환.
        """
        if cls.dim() == 2 and cls.size(1) > 1:
            return cls.argmax(dim=1).long()
        return cls.long()

    @torch.no_grad()
    def _evaluate_one_epoch(self, dataloader: DataLoader, thresh: float = 0.5):
        self.model.eval()

        running = {
            "loss": 0.0,
            "cls_loss": 0.0,
            "cond_loss": 0.0,
            "cls_acc": 0.0,
            "cond_acc": 0.0,
        }
        n_batches = 0

        pbar = tqdm(dataloader, desc="Valid", leave=False)
        for inputs, cls, cond in pbar:
            inputs, cls, cond = self._move_to_device(inputs, cls, cond)
            inputs = self._prepare_inputs(inputs)

            # 타깃 정리
            cls_idx = self._cls_to_index(cls)           # (N,)
            cond = cond.float()                         # (N, 5)  유효 칸만 {0,1}, 나머지는 아무거나여도 mask로 무시

            # forward: 평가에서도 cond는 GT-운동으로 라우팅(teacher forcing)해서 순수 상태 성능을 봄
            logits_cls, logits_cond, mask_cond, _ = self.model(inputs, y_type=cls_idx, mode='train')

            # 손실
            cls_loss = self.CELoss(logits_cls, cls_idx)
            bce = F.binary_cross_entropy_with_logits(logits_cond, cond, reduction='none')  # (N,5)
            cond_loss = (bce * mask_cond.float()).sum() / mask_cond.float().sum().clamp_min(1.0)
            loss = cls_loss + cond_loss

            # 정확도
            cls_pred = logits_cls.argmax(dim=1)
            cls_acc = (cls_pred == cls_idx).float().mean()
            cond_acc = sigmoid_accuracy(logits_cond, cond, thr=thresh, mask=mask_cond)

            # 누적
            running["loss"] += loss.item()
            running["cls_loss"] += cls_loss.item()
            running["cond_loss"] += cond_loss.item()
            running["cls_acc"] += cls_acc.item()
            running["cond_acc"] += cond_acc.item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{running['loss']/n_batches:.4f}",
                "acc":  f"C {running['cls_acc']/n_batches:.3f} / M {running['cond_acc']/n_batches:.3f}",
            })

        for k in running:
            running[k] /= max(1, n_batches)
        return running

    def train(
        self,
        num_epoch: int,
        train_path: str = "tcn_dataset/train",
        val_path: str = "tcn_dataset/val",
        mapper: str = "tcn_dataset/exercise_mapping.json",
        batch_size: int = 8,
        validate: bool = True,
        save_dir: str = "checkpoints",
        cond_thr: float = 0.5,
        lambda_cond: float = 1.0,
        max_grad_norm: float = 2.0,
    ):
        os.makedirs(save_dir, exist_ok=True)
        train_loader = self.make_dataloader(train_path, mapper, batch_size=batch_size, shuffle=True)
        val_loader = self.make_dataloader(val_path, mapper, batch_size=batch_size, shuffle=False) if validate else None

        for epoch in range(1, num_epoch + 1):
            self.model.train()

            running = {
                "loss": 0.0,
                "cls_loss": 0.0,
                "cond_loss": 0.0,
                "cls_acc": 0.0,
                "cond_acc": 0.0,
            }
            n_batches = 0
            pbar = tqdm(train_loader, desc=f"Train {epoch}/{num_epoch}", leave=True)
            for inputs, cls, cond in pbar:
                self.optimizer.zero_grad(set_to_none=True)

                inputs, cls, cond = self._move_to_device(inputs, cls, cond)
                inputs = self._prepare_inputs(inputs)

                cls_idx = self._cls_to_index(cls)   # (N,)
                cond = cond.float()                 # (N,5)  유효 칸만 {0,1}

                # forward: 학습은 GT-운동으로 라우팅(teacher forcing)
                logits_cls, logits_cond, mask_cond, _ = self.model(inputs, y_type=cls_idx, mode='train')

                # Loss Calculate
                cls_loss = self.CELoss(logits_cls, cls_idx)
                bce = F.binary_cross_entropy_with_logits(logits_cond, cond, reduction='none') # (N,5)
                cond_loss = (bce * mask_cond.float()).sum() / mask_cond.float().sum().clamp_min(1.0)
                loss = cls_loss + lambda_cond * cond_loss

                # bwd
                loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()

                # 정확도
                with torch.no_grad():
                    cls_pred = logits_cls.argmax(dim=1)
                    cls_acc = (cls_pred == cls_idx).float().mean()
                    cond_acc = sigmoid_accuracy(logits_cond, cond, thr=cond_thr, mask=mask_cond)

                # 누적
                running["loss"] += loss.item()
                running["cls_loss"] += cls_loss.item()
                running["cond_loss"] += cond_loss.item()
                running["cls_acc"] += cls_acc.item()
                running["cond_acc"] += cond_acc.item()
                n_batches += 1

                pbar.set_postfix({
                    "loss": f"{running['loss']/n_batches:.4f}",
                    "cls":  f"{running['cls_loss']/n_batches:.4f}",
                    "cond": f"{running['cond_loss']/n_batches:.4f}",
                    "acc":  f"C {running['cls_acc']/n_batches:.3f} / M {running['cond_acc']/n_batches:.3f}",
                    "lr":   f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })

            # 스케줄러 한 에폭 스텝
            self.scheduler.step()

            # 에폭 요약
            for k in running:
                running[k] /= max(1, n_batches)
            train_summary = running

            # 검증 (cond는 GT-운동 라우팅 기준)
            if validate and val_loader is not None:
                val_summary = self._evaluate_one_epoch(val_loader, thresh=cond_thr)
            else:
                val_summary = {}

            # 로그 출력
            msg = (
                f"[Epoch {epoch}/{num_epoch}] "
                f"Train loss {train_summary['loss']:.4f} "
                f"(cls {train_summary['cls_loss']:.4f}, cond {train_summary['cond_loss']:.4f}) | "
                f"acc C {train_summary['cls_acc']:.3f} / M {train_summary['cond_acc']:.3f}"
            )
            if val_summary:
                msg += (
                    f" || Val loss {val_summary['loss']:.4f} "
                    f"(cls {val_summary['cls_loss']:.4f}, cond {val_summary['cond_loss']:.4f}) | "
                    f"acc C {val_summary['cls_acc']:.3f} / M {val_summary['cond_acc']:.3f}"
                )
            print(msg)

            # 체크포인트 저장
            ckpt = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            }
            torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch:03d}.pth"))

    def save_ckpt(self, path: str, epoch: int):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path,
        )

    def load_ckpt(self, ckpt_path: str) -> int:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.model.to(self.device)
        print(f"Loaded checkpoint from: {ckpt_path} (epoch={ckpt.get('epoch', '?')})")
        return int(ckpt.get("epoch", 0))


if __name__ == "__main__":
    Base = BaseModel()

    # 예시: 5 epoch 학습
    Base.train(
        num_epoch=5,
        train_path="../tcn_dataset/train",
        val_path="../tcn_dataset/val",
        mapper="../tcn_dataset/exercise_mapping.json",
        batch_size=8,
        validate=True,
        save_dir="checkpoints",
        cond_thr=0.5,
    )

    # 간단 샌티티 체크
    dataset = PoseDataset("../tcn_dataset/val", "../tcn_dataset/exercise_mapping.json")
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=(Base.device == "cuda"),
        drop_last=True,
    )

    Base.model.eval()
    with torch.no_grad():
        for inputs, cls, cond in tqdm(dataloader, desc="Sanity-Check", total=1):
            inputs = inputs.to(Base.device)
            inputs = Base._prepare_inputs(inputs)
            cls_idx = Base._cls_to_index(cls.to(Base.device))
            # 추론: cond는 예측 운동으로 라우팅하려면 mode='eval'
            out_cls, out_cond, mask_cond, extras = Base.model(inputs, y_type=cls_idx, mode='train')
            print("CLS logits:", out_cls.shape, "COND logits:", out_cond.shape, "MASK:", mask_cond.shape)
            break
