import os
import shutil
import json

IMG_W, IMG_H = 1920,1080

ORDER = [
    "Nose","Left Eye","Right Eye","Left Ear","Right Ear",
    "Left Shoulder","Right Shoulder","Left Elbow","Right Elbow",
    "Left Wrist","Right Wrist","Left Hip","Right Hip",
    "Left Knee","Right Knee","Left Ankle","Right Ankle",
    "Neck","Left Palm","Right Palm","Back","Waist",
    "Left Foot","Right Foot"
]

def to_norm(x,y):
    return x/IMG_W, y/IMG_H

if __name__ == '__main__':
    folders = os.listdir("Label")
    for folder in folders:
        data_list = os.listdir(os.path.join("Label",folder))
        for data in data_list:
            if '3d' in data: continue

            f = open(os.path.join("Label",folder,data),encoding='utf-8')
            json_data = json.load(f)
            frames = json_data['frames']

            for frame in frames:
                for view,value in frame.items():
                    #img is alwyas 1920/1080
                    img_path = value['img_key']
                    img_name = img_path.split('/')[-1]
                    shutil.copy(os.path.join("Img",img_path),os.path.join("dataset/images",img_name))

                    pts_dict = value['pts']

                    xs,ys = [],[]
                    kpts = []                    
                    for name in ORDER:
                        x,y = pts_dict[name]['x'], pts_dict[name]['y']
                        xn,yn = to_norm(x,y)
                        kpts += [xn,yn,2]
                        xs.append(x); ys.append(y)
                    
                    x1,x2 = min(xs),max(xs)
                    y1,y2 = min(ys),max(ys)
                    w,h = x2-x1,y2-y1

                    x1 -= w * 0.1; x2 += w * 0.1
                    y1 -= h * 0.1; y2 += h * 0.1
                
                    xc = (x1+x2)/2/IMG_W
                    yc = (y1+y2)/2/IMG_H
                    bw = (x2-x1)/IMG_W
                    bh = (y2-y1)/IMG_H

                    fields = [0,xc,yc,bw,bh] + kpts
                    string = " ".join(str(round(v,10)) for v in fields)
                    txt_name = img_name[:-4] + '.txt'
                    with open(os.path.join("dataset/labels",txt_name),mode='w',encoding='utf-8') as f:
                        f.write(string + "\n")