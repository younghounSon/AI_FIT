import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tcn import TCN

class TCNNetwork(nn.Module):
    def __init__(
        self,
        num_class: int = 41,
        dropout: float = 0.2,
        cond_counts: list[int] | tuple[int, ...] | None = None
    ):
        super(TCNNetwork).__init__()
        self.num_class = num_class
        self.dropout = dropout

        if cond_counts is None: cond_counts = [5] * self.num_class
        assert len(cond_counts) == self.num_class, "Length mismatch with num_class and cond_counts"
        max_conds = max(cond_counts)

        mask_table = torch.zeros(size=(self.num_class,max_conds),dtype=torch.bool)
        for idx,num in enumerate(cond_counts): mask_table[idx,:num] = True
        self.register_buffer("cond_mask_table",mask_table)

        # Backbone to identify Features
        self.backbone = TCN(
            num_inputs=48,
            num_channels=[64,64,64,64],
            kernel_size=3,
            dropout=self.dropout,
            causal=False,
            use_norm='weight_norm',
            activation='relu',
            input_shape='NLC',
            output_projection=None,
            output_activation=None
        )


        self.cls_head = nn.Sequential(
            nn.Linear(64,128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128,self.num_class)
        )

        self.frame_enc = nn.Sequential(
            nn.Conv1d(64,128,kernel_size=1),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.attn_score = nn.Conv1d(128,5,kernel_size=1)
        self.frame_heads = nn.ModuleList([nn.Conv1d(128,5,kernel_size=1) for _ in range(self.num_class)])

    def forward(self, x: torch.tensor, is_train: bool = True):
        feats = self.backbone.forward(x)

        z_cls = feats.mean(dim=1)
        logits_cls = self.cls_head(z_cls)

        h = self.frame_enc(feats.transpose(1,2))
        attn = self.attn_score(h)
        attn = torch.softmax(attn,dim=-1)

        frame_logits_all = torch.stack([head(h) for head in self.frame_heads], dim=1)

        if is_train:
            idx = y


