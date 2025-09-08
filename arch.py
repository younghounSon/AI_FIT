import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tcn import TCN
from typing import List,Optional,Tuple,Union

class TCNModel(nn.Module):
    def __init__(self, num_classes: int = 41, dropout: float = 0.2, cond_counts: list[int] | tuple[int, ...] | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        if cond_counts is None: cond_counts = [5] * num_classes
        assert len(cond_counts) == num_classes
        max_conds = max(cond_counts)

        mask_table = torch.zeros(num_classes, max_conds, dtype=torch.bool)
        for idx,num in enumerate(cond_counts):
            mask_table[idx,:num] = True
        self.register_buffer("cond_valid_table", mask_table)

        # Backbone Definition
        self.backbone = TCN(
            num_inputs=48,
            num_channels=[64, 64, 64, 64],
            kernel_size=3,
            dropout=0.1,
            causal=False,
            use_norm='weight_norm',
            activation='relu',
            input_shape='NLC',
            output_projection=None,
            output_activation=None
        )

        # Class Head Definition
        self.cls_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, num_classes)
        )

        # -------- Condition head (상태) --------
        # 프레임별 특징 인코더: (B,64,T) -> (B,H,T)
        H = 128
        self.frame_enc = nn.Sequential(
            nn.Conv1d(64, H, kernel_size=1),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        # 상태별 어텐션 스코어(공유): (B,H,T) -> (B,5,T)
        self.attn_score = nn.Conv1d(H, 5, kernel_size=1)

        # 운동별 프레임 로짓 head: (B,H,T) -> (B,5,T)
        self.frame_heads = nn.ModuleList([
            nn.Conv1d(H, 5, kernel_size=1) for _ in range(num_classes)
        ])

    def _prep_input(self, x):
        if x.dim() == 4 and x.size(-1) == 2:   # (B,T,K,2) -> (B,T,48)
            B, T, K, _ = x.shape
            x = x.reshape(B, T, K * 2)
        elif x.dim() == 3:
            pass  # already (B,T,48)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        return x

    def forward(self, x, y_type=None, mode: str = 'train'):
        """
        mode: 'train'이면 정답 운동 y_type으로 라우팅(teacher forcing),
              그 외('eval','inference')는 예측 운동으로 라우팅
        """
        x = self._prep_input(x)

        # Backbone: (B,T,48) -> (B,T,64)
        feats = self.backbone(x)          # (B,T,64)
        feats_t = feats.transpose(1, 2)   # (B,64,T) for conv1d

        # 종류 분류: 시간 평균 풀링(필요시 max/std/Δ 추가 가능)
        z_cls = feats.mean(dim=1)         # (B,64)
        logits_cls = self.cls_head(z_cls) # (B,K)

        # 상태: 프레임별 로짓 + 시간 어텐션 풀링
        h = self.frame_enc(feats_t)                   # (B,H,T)
        attn = self.attn_score(h)                     # (B,5,T)
        attn = torch.softmax(attn, dim=-1)            # 상태별 시간분포

        # 모든 운동의 프레임 로짓 계산 후 쌓기: (B,K,5,T)
        frame_logits_all = torch.stack([head(h) for head in self.frame_heads], dim=1)

        # 라우팅: 학습=정답 운동, 추론=예측 운동
        if mode == 'train' and y_type is not None:
            idx = y_type.view(-1, 1, 1, 1).expand(-1, 1, 5, frame_logits_all.size(-1))
            frame_logits = frame_logits_all.gather(1, idx).squeeze(1)     # (B,5,T)
            # 유효 상태 마스크 (B,5)
            mask = self.cond_valid_table[y_type]                          # (B,5) bool
            type_pred = y_type
        else:
            type_pred = logits_cls.argmax(dim=-1)                         # (B,)
            idx = type_pred.view(-1, 1, 1, 1).expand(-1, 1, 5, frame_logits_all.size(-1))
            frame_logits = frame_logits_all.gather(1, idx).squeeze(1)     # (B,5,T)
            mask = self.cond_valid_table[type_pred]                       # (B,5) bool

        # 시간 풀링(어텐션); 필요시 'max'/'topk'로 바꿔 비교 가능
        logits_cond = (frame_logits * attn).sum(dim=-1)                   # (B,5)

        extras = {
            'frame_logits_all': frame_logits_all,  # (B,K,5,T)
            'attn': attn,                          # (B,5, T)
            'type_pred': type_pred                 # (B,)
        }
        return logits_cls, logits_cond, mask, extras


# --------- 사용 예시 (학습 루프) ---------
if __name__ == '__main__':

    B, T, Kpts = 8, 16, 24
    x = torch.rand(B, T, Kpts, 2)          # (B,T,24,2)
    y_type = torch.randint(0, 41, (B,))    # (B,)
    # 각 샘플의 상태 타깃: (B,5). 유효 칸만 0/1, 나머지는 아무 값이어도 mask로 무시됨.
    y_cond = torch.randint(0, 2, (B, 5)).float()

    # 운동별 상태 개수 예: 앞의 2개 운동만 3/4개, 나머지는 5개라고 가정
    cond_counts = [3,4] + [5]*39

    model = TCNModel(num_classes=41, dropout=0.2, cond_counts=cond_counts)
    logits_cls, logits_cond, mask, extras = model(x, y_type=y_type, mode='train')

    # 손실 계산 (마스킹 + 불균형 대응 가능)
    ce = F.cross_entropy(logits_cls, y_type)

    # BCE with mask; 불균형 심하면 BCEWithLogitsLoss(pos_weight=...) 사용
    bce = F.binary_cross_entropy_with_logits(logits_cond, y_cond, reduction='none')  # (B,5)
    cond_loss = (bce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

    loss = ce + 1.0 * cond_loss
    print("cls:", logits_cls.shape)     # (B,41)
    print("cond:", logits_cond.shape)   # (B,5)
    print("mask:", mask.shape)          # (B,5)
    print("loss:", float(loss))
