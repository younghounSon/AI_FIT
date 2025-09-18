import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from model import MultiHeadAGCN  # 기존 model.py 안에 있는 모델 클래스 import

@torch.no_grad()
def evaluate(model, loader, device, num_states_per_ex, meta=None, report_path=None):
    """단일 모델 평가"""
    model.eval()
    total_loss, correct_ex, total_ex = 0.0, 0, 0
    correct_state, total_state = 0, 0
    all_true_ex, all_pred_ex = [], []
    all_true_state, all_pred_state = [], []
    exercise_errors, condition_errors = {}, {}

    exercise_names = meta["exercises"] if meta is not None else None
    exercise_to_conditions = meta["exercise_to_conditions"] if meta is not None else None

    for x, ex_labels, state_labels_per_ex in loader:
        x, ex_labels = x.to(device), ex_labels.to(device)
        state_labels_per_ex = [sl.to(device) for sl in state_labels_per_ex]

        ex_logits, state_logits_list = model(x)
        loss_ex = F.cross_entropy(ex_logits, ex_labels)
        loss_state, batch_size = 0.0, ex_labels.size(0)
        pred_ex = ex_logits.argmax(dim=1)

        all_true_ex.extend(ex_labels.cpu().tolist())
        all_pred_ex.extend(pred_ex.cpu().tolist())

        for i in range(batch_size):
            true_ex, pred_ex_i = ex_labels[i].item(), pred_ex[i].item()
            if pred_ex_i == true_ex:
                logits_i = state_logits_list[true_ex][i].unsqueeze(0)
                targets_i = state_labels_per_ex[true_ex][i].unsqueeze(0)
                loss_state += F.binary_cross_entropy_with_logits(logits_i, targets_i)
                preds = (torch.sigmoid(logits_i) > 0.5).int()
                all_true_state.extend(targets_i.cpu().numpy().flatten().tolist())
                all_pred_state.extend(preds.cpu().numpy().flatten().tolist())
                if torch.equal(preds, targets_i.int()):
                    correct_state += 1
                else:
                    if meta is not None:
                        ex_name, cond_names = exercise_names[true_ex], exercise_to_conditions[exercise_names[true_ex]]
                        wrong_mask = preds[0] != targets_i[0].int()
                        for j, wrong in enumerate(wrong_mask):
                            if wrong:
                                cond = cond_names[j]
                                condition_errors.setdefault(ex_name, {})
                                condition_errors[ex_name][cond] = condition_errors[ex_name].get(cond, 0) + 1
                total_state += 1
            else:
                total_state += 1
                if meta is not None:
                    key = (exercise_names[true_ex], exercise_names[pred_ex_i])
                    exercise_errors[key] = exercise_errors.get(key, 0) + 1

        total_loss += (loss_ex + loss_state).item() * batch_size
        correct_ex += (pred_ex == ex_labels).sum().item()
        total_ex += batch_size

    acc_ex = correct_ex / total_ex
    acc_state = correct_state / total_state if total_state > 0 else 0.0
    prec_ex, rec_ex, f1_ex, _ = precision_recall_fscore_support(all_true_ex, all_pred_ex, average="macro", zero_division=0)
    prec_state, rec_state, f1_state, _ = precision_recall_fscore_support(all_true_state, all_pred_state, average="micro", zero_division=0)

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
        "acc_ex": acc_ex, "acc_state": acc_state,
        "prec_ex": prec_ex, "rec_ex": rec_ex, "f1_ex": f1_ex,
        "prec_state": prec_state, "rec_state": rec_state, "f1_state": f1_state
    }

@torch.no_grad()
def evaluate_ensemble(model_joint, model_bone, loader_joint, loader_bone, device,
                      num_states_per_ex, meta=None, report_path=None, alpha=0.5):
    """joint + bone 앙상블 평가 (joint/bone 로더 분리)"""
    model_joint.eval()
    model_bone.eval()

    all_true_ex, all_pred_ex = [], []
    all_true_state, all_pred_state = [], []
    total_loss, correct_ex, total_ex, correct_state, total_state = 0, 0, 0, 0, 0
    exercise_errors, condition_errors = {}, {}

    exercise_names = meta["exercises"] if meta is not None else None
    exercise_to_conditions = meta["exercise_to_conditions"] if meta is not None else None

    # joint와 bone 로더를 zip으로 묶어 동기화
    for (x_j, ex_labels_j, state_labels_j), (x_b, _, state_labels_b) in zip(loader_joint, loader_bone):
        x_j, ex_labels_j = x_j.to(device), ex_labels_j.to(device)
        x_b = x_b.to(device)
        state_labels_j = [sl.to(device) for sl in state_labels_j]

        # 모델별 forward
        ex_logits_j, state_logits_j = model_joint(x_j)
        ex_logits_b, state_logits_b = model_bone(x_b)

        # 앙상블 (가중합)
        ex_logits = alpha * ex_logits_j + (1 - alpha) * ex_logits_b
        state_logits_list = [
            alpha * sj + (1 - alpha) * sb
            for sj, sb in zip(state_logits_j, state_logits_b)
        ]

        loss_ex = F.cross_entropy(ex_logits, ex_labels_j)
        loss_state, batch_size = 0.0, ex_labels_j.size(0)
        pred_ex = ex_logits.argmax(dim=1)

        all_true_ex.extend(ex_labels_j.cpu().tolist())
        all_pred_ex.extend(pred_ex.cpu().tolist())

        for i in range(batch_size):
            true_ex, pred_ex_i = ex_labels_j[i].item(), pred_ex[i].item()
            if pred_ex_i == true_ex:
                logits_i = state_logits_list[true_ex][i].unsqueeze(0)
                targets_i = state_labels_j[true_ex][i].unsqueeze(0)
                loss_state += F.binary_cross_entropy_with_logits(logits_i, targets_i)
                preds = (torch.sigmoid(logits_i) > 0.5).int()
                all_true_state.extend(targets_i.cpu().numpy().flatten().tolist())
                all_pred_state.extend(preds.cpu().numpy().flatten().tolist())
                if torch.equal(preds, targets_i.int()):
                    correct_state += 1
                else:
                    if meta is not None:
                        ex_name, cond_names = exercise_names[true_ex], exercise_to_conditions[exercise_names[true_ex]]
                        wrong_mask = preds[0] != targets_i[0].int()
                        for j, wrong in enumerate(wrong_mask):
                            if wrong:
                                cond = cond_names[j]
                                condition_errors.setdefault(ex_name, {})
                                condition_errors[ex_name][cond] = condition_errors[ex_name].get(cond, 0) + 1
                total_state += 1
            else:
                total_state += 1
                if meta is not None:
                    key = (exercise_names[true_ex], exercise_names[pred_ex_i])
                    exercise_errors[key] = exercise_errors.get(key, 0) + 1

        total_loss += (loss_ex + loss_state).item() * batch_size
        correct_ex += (pred_ex == ex_labels_j).sum().item()
        total_ex += batch_size

    acc_ex = correct_ex / total_ex
    acc_state = correct_state / total_state if total_state > 0 else 0.0
    prec_ex, rec_ex, f1_ex, _ = precision_recall_fscore_support(all_true_ex, all_pred_ex, average="macro", zero_division=0)
    prec_state, rec_state, f1_state, _ = precision_recall_fscore_support(all_true_state, all_pred_state, average="micro", zero_division=0)

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
        "acc_ex": acc_ex, "acc_state": acc_state,
        "prec_ex": prec_ex, "rec_ex": rec_ex, "f1_ex": f1_ex,
        "prec_state": prec_state, "rec_state": rec_state, "f1_state": f1_state
    }


if __name__ == "__main__":
    import pickle
    from torch.utils.data import DataLoader
    from dataset import ExerciseDataset, collate_fn

    device = "cuda"
    tag_joint, tag_bone = "joint", "bone"
    num_point, num_person = 24, 1
    in_channels, drop_out = 2, 0.5

    # ----- 메타데이터 -----
    tmp = pickle.load(open("val_data.pkl", "rb"))
    meta = tmp["meta"]
    exercises = meta["exercises"]
    num_exercises = len(exercises)
    num_states_per_ex = [len(meta["exercise_to_conditions"][ex]) for ex in exercises]

    val_dataset = ExerciseDataset("val_data.pkl", use_bone=False)
    val_dataset_joint = ExerciseDataset("val_data.pkl", use_bone=False)
    val_loader_joint = DataLoader(
        val_dataset_joint, batch_size=32, shuffle=False,
        collate_fn=lambda b: collate_fn(b, num_states_per_ex)
    )

    val_dataset_bone = ExerciseDataset("val_data.pkl", use_bone=True)
    val_loader_bone = DataLoader(
        val_dataset_bone, batch_size=32, shuffle=False,
        collate_fn=lambda b: collate_fn(b, num_states_per_ex)
    )

    # ----- 모델 준비 -----
    model_joint = MultiHeadAGCN(num_exercises, num_states_per_ex, num_point, num_person,
                                graph="graph.mygraph.Graph", graph_args={"labeling_mode": "spatial"},
                                in_channels=in_channels, drop_out=drop_out).to(device)
    model_joint.load_state_dict(torch.load(f"best_{tag_joint}.pth"))

    model_bone = MultiHeadAGCN(num_exercises, num_states_per_ex, num_point, num_person,
                               graph="graph.mygraph.Graph", graph_args={"labeling_mode": "spatial"},
                               in_channels=in_channels, drop_out=drop_out).to(device)
    ckpt = torch.load("checkpoint_bone_epoch60.pth", map_location="cpu")
    model_bone.load_state_dict(ckpt["model"])
    #model_bone.load_state_dict(torch.load(f"checkpoint_bone_epoch60.pth"))

    # ----- 데이터셋 & 로더 -----
    val_dataset_joint = ExerciseDataset("val_data.pkl", use_bone=False)
    val_loader_joint = DataLoader(val_dataset_joint, batch_size=32, shuffle=False,
                                collate_fn=lambda b: collate_fn(b, num_states_per_ex))

    val_dataset_bone = ExerciseDataset("val_data.pkl", use_bone=True)
    val_loader_bone = DataLoader(val_dataset_bone, batch_size=32, shuffle=False,
                                collate_fn=lambda b: collate_fn(b, num_states_per_ex))

    # ----- 평가 -----
    m_joint = evaluate(model_joint, val_loader_joint, device, num_states_per_ex, meta, report_path="eval_report_joint.txt")
    m_bone = evaluate(model_bone, val_loader_bone, device, num_states_per_ex, meta, report_path="eval_report_bone.txt")
    m_ens = evaluate_ensemble(model_joint, model_bone, val_loader_joint, val_loader_bone, device,
                            num_states_per_ex, meta, report_path="eval_report_ensemble.txt", alpha=0.5)


    print("\n✅ Evaluation Results")
    print("Joint:")
    print(f"  ExAcc {m_joint['acc_ex']:.4f}, StateAcc {m_joint['acc_state']:.4f}, "
          f"ExF1 {m_joint['f1_ex']:.4f}, StateF1 {m_joint['f1_state']:.4f}")
    print("Bone:")
    print(f"  ExAcc {m_bone['acc_ex']:.4f}, StateAcc {m_bone['acc_state']:.4f}, "
          f"ExF1 {m_bone['f1_ex']:.4f}, StateF1 {m_bone['f1_state']:.4f}")
    print("Ensemble:")
    print(f"  ExAcc {m_ens['acc_ex']:.4f}, StateAcc {m_ens['acc_state']:.4f}, "
          f"ExF1 {m_ens['f1_ex']:.4f}, StateF1 {m_ens['f1_state']:.4f}")
