import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from model_action.arch import MultiHeadAGCN
from model_action.dataset import ExerciseDataset

def pad_or_trim(x, target_len=16):
    C, T, V, M = x.shape
    if T == target_len:
        return x
    elif T < target_len:
        # pad 마지막 프레임 반복
        pad = x[:, -1:, :, :].repeat(1, target_len-T, 1, 1)
        return torch.cat([x, pad], dim=1)
    else:
        # 잘라내기 (앞부분만 사용)
        return x[:, :target_len, :, :]

def collate_fn(batch, num_states_per_ex, target_len=16):
    xs, ex_labels, conds = zip(*batch)
    xs = [pad_or_trim(x, target_len) for x in xs]
    xs = torch.stack(xs)  # [N,C,T,V,M]
    ex_labels = torch.stack(ex_labels)

    state_labels_per_ex = []
    for ex_idx, n_state in enumerate(num_states_per_ex):
        tmp = torch.zeros(len(batch), n_state, dtype=torch.float32)
        for i, (lbl, cond_vec) in enumerate(zip(ex_labels, conds)):
            if lbl.item() == ex_idx:
                tmp[i] = cond_vec
        state_labels_per_ex.append(tmp)
    return xs, ex_labels, state_labels_per_ex


# -----------------------------
# Train / Eval functions
# -----------------------------

class AGCNModel():
    def __init__(self,device='cuda'):
        self.model = MultiHeadAGCN(
            num_exercises=51,
            num_states_per_exercise=num_states_per_ex,
            num_point=24,
            num_person=5,
            graph="graph.mygraph.Graph",
            graph_args={"labeling_mode": "spatial"},
            in_channels=2,  # x,y만 쓰는 경우
            drop_out=0.5
        ).cuda()
        print("Hello")


    def train_one_epoch(self,model, loader, optimizer, device, num_states_per_ex):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, ex_labels, state_labels_per_ex in loader:
            x, ex_labels = x.to(device), ex_labels.to(device)
            state_labels_per_ex = [sl.to(device) for sl in state_labels_per_ex]

            optimizer.zero_grad()
            ex_logits, state_logits_list = model(x)

            # 운동 종류 loss
            loss_ex = F.cross_entropy(ex_logits, ex_labels)

            # 상태 loss
            loss_state = 0.0
            for ex_idx, (logits_i, targets_i) in enumerate(zip(state_logits_list, state_labels_per_ex)):
                mask = (ex_labels == ex_idx)
                if mask.any():
                    li = logits_i[mask]
                    ti = targets_i[mask]
                    loss_state += F.binary_cross_entropy_with_logits(li, ti)

            loss = loss_ex + loss_state
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = ex_logits.argmax(dim=1)
            correct += (pred == ex_labels).sum().item()
            total += ex_labels.size(0)

        return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_states_per_ex):
    model.eval()
    total_loss, correct_ex, total_ex = 0.0, 0, 0

    # 상태 정확도 계산용
    correct_state, total_state = 0, 0

    for x, ex_labels, state_labels_per_ex in loader:
        x, ex_labels = x.to(device), ex_labels.to(device)
        state_labels_per_ex = [sl.to(device) for sl in state_labels_per_ex]

        ex_logits, state_logits_list = model(x)

        # 운동 종류 loss
        loss_ex = F.cross_entropy(ex_logits, ex_labels)

        # 상태 loss + 상태 정확도
        loss_state = 0.0
        for ex_idx, (logits_i, targets_i) in enumerate(zip(state_logits_list, state_labels_per_ex)):
            mask = (ex_labels == ex_idx)
            if mask.any():
                li = logits_i[mask]
                ti = targets_i[mask]
                loss_state += F.binary_cross_entropy_with_logits(li, ti)

                # 상태 accuracy (배치 단위에서 바로 누적)
                probs = torch.sigmoid(li)
                preds = (probs > 0.5).int()
                correct_state += (preds == ti.int()).sum().item()
                total_state += preds.numel()

        loss = loss_ex + loss_state
        total_loss += loss.item() * x.size(0)

        # 운동 종류 accuracy
        pred_ex = ex_logits.argmax(dim=1)
        correct_ex += (pred_ex == ex_labels).sum().item()
        total_ex += ex_labels.size(0)

    acc_ex = correct_ex / total_ex
    acc_state = correct_state / total_state if total_state > 0 else 0.0

    return total_loss / total_ex, acc_ex, acc_state



# -----------------------------
# Main
# -----------------------------
def main():
    # 경로 지정
    train_pkl = "train.pkl"
    val_pkl = "val.pkl"
    use_bone = False  # True면 bone stream 학습, False면 joint stream 학습

    # Dataset & Meta 불러오기
    tmp = pickle.load(open(train_pkl, "rb"))
    meta = tmp["meta"]
    exercises = meta["exercises"]
    num_exercises = len(exercises)
    num_states_per_ex = [len(meta["exercise_to_conditions"][ex]) for ex in exercises]

    train_dataset = ExerciseDataset(train_pkl, use_bone=use_bone)
    val_dataset = ExerciseDataset(val_pkl, use_bone=use_bone)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        collate_fn=lambda b: collate_fn(b, num_states_per_ex)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        collate_fn=lambda b: collate_fn(b, num_states_per_ex)
    )

    # 모델 생성
    model = MultiHeadAGCN(
        num_exercises=num_exercises,
        num_states_per_exercise=num_states_per_ex,
        num_point=24,
        num_person=5,
        graph="graph.mygraph.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=2,  # x,y만 쓰는 경우
        drop_out=0.5
    ).cuda()

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=[45, 55], gamma=0.1)
    num_epochs = 65
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, "cuda", num_states_per_ex)
        val_loss, val_ex_acc, val_state_acc = evaluate(model, val_loader, "cuda", num_states_per_ex)
        scheduler.step()

        print(f"[{epoch:03d}] "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} ExAcc {val_ex_acc:.4f} StateAcc {val_state_acc:.4f}")

        # 모델 저장
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pth")


if __name__ == "__main__":
    main()
