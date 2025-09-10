import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

HEIGHT = 1080
WIDTH = 1920
ORDER = [
    "Nose","Left Eye","Right Eye","Left Ear","Right Ear",
    "Left Shoulder","Right Shoulder","Left Elbow","Right Elbow",
    "Left Wrist","Right Wrist","Left Hip","Right Hip",
    "Left Knee","Right Knee","Left Ankle","Right Ankle",
    "Neck","Left Palm","Right Palm","Back","Waist",
    "Left Foot","Right Foot"
]

# flip_idx = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,17,19,18,20,21,23,22]

class PoseDataset(Dataset):
    def __init__(self,file_dir,mapping):
        super().__init__()
        self.file_list = [os.path.join(file_dir, fn) for fn in os.listdir(file_dir) if fn.endswith(".json")]
        with open(mapping,"r",encoding="utf-8") as f: self.mapper = json.load(f)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self,idx):
        path = self.file_list[idx]
        with open(path,'r',encoding='utf-8') as f: data = json.load(f)
        frames = data['frames']; type_info = data['type_info']
        exercise_type = type_info['exercise']
        exercise_cond = type_info['conditions']
        cond_value = [cond["value"] for cond in exercise_cond]
        while len(cond_value) < 5: cond_value.append(False)

        # input data 
        xy = np.zeros((len(frames),len(ORDER),2),dtype=np.float32)
        for t,frame in enumerate(frames):
            for k,part in enumerate(ORDER):
                x,y = frame[part]['x'], frame[part]['y']
                xn, yn = x/WIDTH, y/HEIGHT
                xy[t,k,0] = xn; xy[t,k,1] = yn
        
        # output_class data
        cls_label = np.zeros(len(self.mapper),dtype=np.float32)
        cls_label[self.mapper[exercise_type]] = 1

        # output_cond data
        cond_label = np.asarray(cond_value,dtype=np.float32)
        
        return xy,cls_label,cond_label

if __name__ == '__main__':
    dataset_val = PoseDataset("./tcn_dataset/train", mapping="./tcn_dataset/exercise_mapping.json")
    dataloader_val = DataLoader(dataset_val, batch_size=8, shuffle=True, num_workers=0)

    true_count = np.zeros(5, dtype=np.int64)
    total_count = np.zeros(5, dtype=np.int64)

    for inputs, cls, cond in tqdm(dataloader_val):
        # cond: (B,5) float tensor
        cond_np = cond.numpy().astype(bool)  # (B,5)
        true_count += cond_np.sum(axis=0)
        total_count += cond_np.shape[0]      # batch 크기만큼 각 상태에 누적

    ratios = true_count / total_count

    print("=== Condition True Ratio per index ===")
    for i, r in enumerate(ratios):
        print(f"Cond {i}: True {true_count[i]} / {total_count[i]} "
              f"({r*100:.2f}% True, {(1-r)*100:.2f}% False)")

    
