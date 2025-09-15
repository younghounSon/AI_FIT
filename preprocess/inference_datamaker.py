import os
import json
import shutil
import copy
LABEL_PATH = "S:/FIT/Label2"
VIEW = ['view1','view2','view3','view4','view5']

def remove_abnormal():
    lists = os.listdir(LABEL_PATH)
    for file in lists:
        with open(os.path.join(LABEL_PATH,file),"r",encoding="utf-8") as f:
            data = json.load(f)

        frames = data.get('frames',[])

        if len(frames) > 16:
            print(f"Too many : {file}")
            data['frames'] = frames[:16]
            assert len(data['frames']) == 16
            
            with open(os.path.join(LABEL_PATH,file),"w",encoding="utf-8") as f:
                json.dump(data,f,indent=4,ensure_ascii=False)
        elif len(frames) < 16:
            print(f"Too small : {file}")
            os.remove(os.path.join(LABEL_PATH,file))

def split():
    lists = os.listdir(LABEL_PATH)
    for file in lists:
        if 'view' in file: continue
        with open(os.path.join(LABEL_PATH,file),"r",encoding="utf-8") as f:
            data = json.load(f)

        frames = data.get('frames',[])
        for view in VIEW:
            new_frame = []
            for frame in frames: new_frame.append(frame[view])
            new_data = copy.deepcopy(data)
            new_data['frames'] = new_frame
            
            out_name = f"{os.path.splitext(file)[0]}_{view}.json"
            with open(os.path.join(LABEL_PATH,out_name),"w",encoding="utf-8") as f:
                json.dump(new_data,f,indent=4,ensure_ascii=False)

def remove_original():
    lists = os.listdir(LABEL_PATH)
    for file in lists:
        if "view" in file: continue
        os.remove(os.path.join(LABEL_PATH,file))
        print(f"Removed : {file}")

if __name__ == '__main__':
    remove_abnormal()
    split()
    remove_original()