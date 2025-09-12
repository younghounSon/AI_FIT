import pickle
import torch
from torch.utils.data import Dataset

class ExerciseDataset(Dataset):
    def __init__(self, pickel_path: str, use_bone: bool = False):
        data = pickle.load(open(pickel_path,"rb"))
        self.samples = data["samples"]
        self.meta = data["meta"]
        self.use_bone = use_bone
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        joint, bone, ex_label, cond_vec = self.samples[idx]
        x = bone if self.use_bone else joint
        return (
            torch.tensor(x,dtype=torch.float32),
            torch.tensor(ex_label, dtype=torch.long),
            torch.tensor(cond_vec, dtype=torch.float32)
        )
    
if __name__ == '__main__':
    print("This is ExerciseDataset.")
    exit()