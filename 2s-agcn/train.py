import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from model import MultiHeadAGCN  # 멀티헤드 모델
from sklearn.metrics import precision_recall_fscore_support

# -----------------------------
# Dataset
# -----------------------------
class ExerciseDataset(Dataset):
    def __init__(self, pickle_path, use_bone=False):
        data = pickle.load(open(pickle_path, "rb"))
        self.samples = data["samples"]
        self.meta = data["meta"]
        self.use_bone = use_bone

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        joint, bone, ex_label, cond_vec = self.samples[idx]
        x = bone if self.use_bone else joint
        return (
            torch.tensor(x, dtype=torch.float32),   # [C,T,V,M]
            torch.tensor(ex_label, dtype=torch.long),
            torch.tensor(cond_vec, dtype=torch.float32)
        )


# -----------------------------
# Collate Function
# -----------------------------

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
def train_one_epoch(model, loader, optimizer, device, num_states_per_ex):
    model.train()
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

"""
@torch.no_grad()
def evaluate(model, loader, device, num_states_per_ex, meta=None, report_path=None):
    model.eval()
    total_loss, correct_ex, total_ex = 0.0, 0, 0
    correct_state, total_state = 0, 0

    # 리포트용 통계
    exercise_errors = {}   # { (true_ex, pred_ex): count }
    condition_errors = {}  # { exercise_name: {condition: count} }

    exercise_names = meta["exercises"] if meta is not None else None
    exercise_to_conditions = meta["exercise_to_conditions"] if meta is not None else None

    for x, ex_labels, state_labels_per_ex in loader:
        x, ex_labels = x.to(device), ex_labels.to(device)
        state_labels_per_ex = [sl.to(device) for sl in state_labels_per_ex]

        ex_logits, state_logits_list = model(x)
        loss_ex = F.cross_entropy(ex_logits, ex_labels)

        loss_state = 0.0
        batch_size = ex_labels.size(0)

        pred_ex = ex_logits.argmax(dim=1)

        for i in range(batch_size):
            true_ex = ex_labels[i].item()
            pred_ex_i = pred_ex[i].item()

            if pred_ex_i == true_ex:
                logits_i = state_logits_list[true_ex][i].unsqueeze(0)  # [1, cond_dim]
                targets_i = state_labels_per_ex[true_ex][i].unsqueeze(0)

                loss_state += F.binary_cross_entropy_with_logits(logits_i, targets_i)

                probs = torch.sigmoid(logits_i)
                preds = (probs > 0.5).int()
                if torch.equal(preds, targets_i.int()):
                    correct_state += 1
                else:
                    # 어떤 상태 조건이 틀렸는지 기록
                    if meta is not None:
                        ex_name = exercise_names[true_ex]
                        cond_names = exercise_to_conditions[ex_name]
                        wrong_mask = preds[0] != targets_i[0].int()
                        for j, wrong in enumerate(wrong_mask):
                            if wrong:
                                cond = cond_names[j]
                                if ex_name not in condition_errors:
                                    condition_errors[ex_name] = {}
                                condition_errors[ex_name][cond] = (
                                    condition_errors[ex_name].get(cond, 0) + 1
                                )
                total_state += 1
            else:
                total_state += 1
                if meta is not None:
                    true_ex_name = exercise_names[true_ex]
                    pred_ex_name = exercise_names[pred_ex_i]
                    key = (true_ex_name, pred_ex_name)
                    exercise_errors[key] = exercise_errors.get(key, 0) + 1

        loss = loss_ex + loss_state
        total_loss += loss.item() * batch_size
        correct_ex += (pred_ex == ex_labels).sum().item()
        total_ex += batch_size

    acc_ex = correct_ex / total_ex
    acc_state = correct_state / total_state if total_state > 0 else 0.0

    # 리포트 저장 (최종 한 번만 호출하면 됨)
    if report_path and meta is not None:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("📊 Exercise Misclassifications:\n")
            for (true_ex, pred_ex), count in exercise_errors.items():
                f.write(f" - {true_ex} → {pred_ex}: {count} times\n")

            f.write("\n📊 Condition Errors (per exercise):\n")
            for ex, cond_dict in condition_errors.items():
                f.write(f" - {ex}:\n")
                for cond, count in cond_dict.items():
                    f.write(f"    * {cond}: {count} times\n")

    return total_loss / total_ex, acc_ex, acc_state
"""
@torch.no_grad()
def evaluate(model, loader, device, num_states_per_ex, meta=None, report_path=None):
    model.eval()
    total_loss, correct_ex, total_ex = 0.0, 0, 0
    correct_state, total_state = 0, 0

    # 전체 예측/정답 누적 (metrics 계산용)
    all_true_ex, all_pred_ex = [], []
    all_true_state, all_pred_state = [], []  # flatten된 멀티라벨

    # 리포트용 통계
    exercise_errors = {}
    condition_errors = {}

    exercise_names = meta["exercises"] if meta is not None else None
    exercise_to_conditions = meta["exercise_to_conditions"] if meta is not None else None

    for x, ex_labels, state_labels_per_ex in loader:
        x, ex_labels = x.to(device), ex_labels.to(device)
        state_labels_per_ex = [sl.to(device) for sl in state_labels_per_ex]

        ex_logits, state_logits_list = model(x)
        loss_ex = F.cross_entropy(ex_logits, ex_labels)

        loss_state = 0.0
        batch_size = ex_labels.size(0)

        pred_ex = ex_logits.argmax(dim=1)

        # ---- exercise 기록 ----
        all_true_ex.extend(ex_labels.cpu().tolist())
        all_pred_ex.extend(pred_ex.cpu().tolist())

        for i in range(batch_size):
            true_ex = ex_labels[i].item()
            pred_ex_i = pred_ex[i].item()

            if pred_ex_i == true_ex:
                logits_i = state_logits_list[true_ex][i].unsqueeze(0)
                targets_i = state_labels_per_ex[true_ex][i].unsqueeze(0)

                loss_state += F.binary_cross_entropy_with_logits(logits_i, targets_i)

                probs = torch.sigmoid(logits_i)
                preds = (probs > 0.5).int()

                # flatten 결과 저장
                all_true_state.extend(targets_i.cpu().numpy().flatten().tolist())
                all_pred_state.extend(preds.cpu().numpy().flatten().tolist())

                if torch.equal(preds, targets_i.int()):
                    correct_state += 1
                else:
                    if meta is not None:
                        ex_name = exercise_names[true_ex]
                        cond_names = exercise_to_conditions[ex_name]
                        wrong_mask = preds[0] != targets_i[0].int()
                        for j, wrong in enumerate(wrong_mask):
                            if wrong:
                                cond = cond_names[j]
                                if ex_name not in condition_errors:
                                    condition_errors[ex_name] = {}
                                condition_errors[ex_name][cond] = (
                                    condition_errors[ex_name].get(cond, 0) + 1
                                )
                total_state += 1
            else:
                total_state += 1
                if meta is not None:
                    true_ex_name = exercise_names[true_ex]
                    pred_ex_name = exercise_names[pred_ex_i]
                    key = (true_ex_name, pred_ex_name)
                    exercise_errors[key] = exercise_errors.get(key, 0) + 1

        loss = loss_ex + loss_state
        total_loss += loss.item() * batch_size
        correct_ex += (pred_ex == ex_labels).sum().item()
        total_ex += batch_size

    # ---- accuracy ----
    acc_ex = correct_ex / total_ex
    acc_state = correct_state / total_state if total_state > 0 else 0.0

    # ---- precision / recall / f1 ----
    # Exercise (multi-class)
    prec_ex, rec_ex, f1_ex, _ = precision_recall_fscore_support(
        all_true_ex, all_pred_ex, average="macro", zero_division=0
    )

    # State (multi-label → flatten binary)
    prec_state, rec_state, f1_state, _ = precision_recall_fscore_support(
        all_true_state, all_pred_state, average="micro", zero_division=0
    )

    # ---- 리포트 저장 ----
    if report_path and meta is not None:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("📊 Exercise Misclassifications:\n")
            for (true_ex, pred_ex), count in exercise_errors.items():
                f.write(f" - {true_ex} → {pred_ex}: {count} times\n")

            f.write("\n📊 Condition Errors (per exercise):\n")
            for ex, cond_dict in condition_errors.items():
                f.write(f" - {ex}:\n")
                for cond, count in cond_dict.items():
                    f.write(f"    * {cond}: {count} times\n")

    return {
        "loss": total_loss / total_ex,
        "acc_ex": acc_ex,
        "acc_state": acc_state,
        "prec_ex": prec_ex, "rec_ex": rec_ex, "f1_ex": f1_ex,
        "prec_state": prec_state, "rec_state": rec_state, "f1_state": f1_state
    }



def run_training(train_pkl, val_pkl, use_bone, tag="joint", resume_ckpt=None, start_epoch=1):
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

    model = MultiHeadAGCN(
        num_exercises=num_exercises,
        num_states_per_exercise=num_states_per_ex,
        num_point=24,
        num_person=1,
        graph="graph.mygraph.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=2,
        drop_out=0.5
    ).cuda()

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[45, 55], gamma=0.1)

    # ✅ 이어 학습
    if resume_ckpt:
        model.load_state_dict(torch.load(resume_ckpt))
        #ckpt = torch.load(resume_ckpt)
        #model.load_state_dict(ckpt["model"])
        #optimizer.load_state_dict(ckpt["optimizer"])
        #scheduler.load_state_dict(ckpt["scheduler"])
        #start_epoch = ckpt["epoch"] + 1
        #print(f"🔄 Resume training from {resume_ckpt} (epoch {start_epoch})")
        

    best_acc = 0.0
    patience = 10
    wait = 0
    num_epochs = 200

    for epoch in range(start_epoch, num_epochs + 1):
    # ---- Train ----
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, "cuda", num_states_per_ex)

        # ---- Validation ----
        val_metrics = evaluate(model, val_loader, "cuda", num_states_per_ex, meta=None)  
        # 반환: dict
        # {
        #   "loss", "acc_ex", "acc_state",
        #   "prec_ex", "rec_ex", "f1_ex",
        #   "prec_state", "rec_state", "f1_state"
        # }

        scheduler.step()

        # ---- 로그 문자열 구성 ----
        log_str = (
            f"[{tag}][{epoch:03d}] "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} "
            f"ExAcc {val_metrics['acc_ex']:.4f} "
            f"StateAcc {val_metrics['acc_state']:.4f} | "
            f"Ex(P/R/F1) {val_metrics['prec_ex']:.3f}/{val_metrics['rec_ex']:.3f}/{val_metrics['f1_ex']:.3f} "
            f"State(P/R/F1) {val_metrics['prec_state']:.3f}/{val_metrics['rec_state']:.3f}/{val_metrics['f1_state']:.3f}"
        )
        print(log_str)
        with open(f"train_log_{tag}.txt", "a", encoding="utf-8") as f:
            f.write(log_str + "\n")

        # ---- Best 저장 (state accuracy 기준) ----
        if val_metrics["acc_state"] > best_acc:
            best_acc = val_metrics["acc_state"]
            torch.save(model.state_dict(), f"best_{tag}.pth")
            print(f"✅ Best {tag} model updated at epoch {epoch} (StateAcc={best_acc:.4f})")
            wait = 0
        else:
            wait += 1
            print(f"⏳ No improvement for {wait}/{patience} epochs")

        # ---- 체크포인트 저장 ----
        if epoch % 10 == 0:
            ckpt_path = f"checkpoint_{tag}_epoch{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

        if wait >= patience:
            print(f"⛔ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # ---- Last 저장 ----
    torch.save(model.state_dict(), f"last_{tag}.pth")

    # ---- 최종 리포트 (best model 불러와서 평가) ----
    print("\n🔎 Saving final evaluation report (best model)...")
    best_model = MultiHeadAGCN(
        num_exercises=num_exercises,
        num_states_per_exercise=num_states_per_ex,
        num_point=24,
        num_person=1,
        graph="graph.mygraph.Graph",
        graph_args={"labeling_mode": "spatial"},
        in_channels=2,
        drop_out=0.5
    ).cuda()
    best_model.load_state_dict(torch.load(f"best_{tag}.pth"))

    final_metrics = evaluate(best_model, val_loader, "cuda", num_states_per_ex, meta,
                            report_path=f"eval_report_{tag}.txt")

    print("✅ Final Metrics:",
        f"ExAcc {final_metrics['acc_ex']:.4f}, "
        f"StateAcc {final_metrics['acc_state']:.4f}, "
        f"Ex F1 {final_metrics['f1_ex']:.4f}, "
        f"State F1 {final_metrics['f1_state']:.4f}")
    print(f"📄 Final report saved to eval_report_{tag}.txt")


# -----------------------------
# Main
# -----------------------------

def main():
    train_pkl = "train_data.pkl"
    val_pkl = "val_data.pkl"
    
    # 2️⃣ Bone stream 학습
    run_training(train_pkl, val_pkl, use_bone=True, tag="bone")
    #1️⃣ Joint stream 학습
    run_training(train_pkl, val_pkl, use_bone=False, tag="joint")
    
    
    

    


if __name__ == "__main__":
    main()

