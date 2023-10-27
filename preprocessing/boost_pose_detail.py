from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
import torch
import os
import os.path
import cv2 as cv
import numpy as np
import torch


def walk_dir(root:os.PathLike)->list:
    u = []
    for r, dirs, files in os.walk(root, topdown=False):
        for f in files:
            u.append(os.path.join(r, f))             
    return u


def detection(img_name:os.PathLike, model_pose:YOLO, model_seg:YOLO, out_name:os.PathLike):
    i=cv.imread(img_name)#get origin image
    i = cv.resize(i, (640,640))
    results_seg = model_seg(i, verbose=False)#image name
    #print(img_name)
    for r in results_seg:
        cls = r.boxes.cls.to(dtype=torch.int32)
        j = torch.argwhere(cls == 0)
        person_cls = -1
        try:
            person_cls = j[0]
        except:
            continue
        a_person = person_cls[0].item()
        m = (r.masks.data)[a_person]
        mask = torch.permute(m[None, :, :], (1, 2, 0)).numpy()
        mask_resize = cv.resize(mask, (i.shape[1], i.shape[0]))
        img = np.copy(i)
        img[:,:,0] = img[:,:,0]*mask_resize
        img[:,:,1] = img[:,:,1]*mask_resize
        img[:,:,2] = img[:,:,2]*mask_resize
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray2 = cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)
        gray3 = cv.Canny(gray2, 125, 175)
        cnts, _ = cv.findContours(gray.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cnts2, _ = cv.findContours(gray3.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        clone = img.copy()*0
        cv.drawContours(clone, cnts, -1, (255, 255, 255), 2)
        cv.drawContours(clone, cnts2, -1, (255, 255, 255), 2)
        results_pose = model_pose(img, verbose=False)
        im_array = results_pose[0].plot(boxes=False, labels=False, img=clone)
        bbox = results_pose[0].boxes.xyxy.squeeze()
        bbox = bbox.to(dtype=int).tolist()
        if len(bbox) < 4:
            print(img_name)
            continue
        im_array = im_array[bbox[1]:bbox[3], bbox[0]:bbox[2],:]
        cv.imwrite(out_name,im_array)

def main(models:dict, imgs_path:list):
    for imgpath in tqdm(imgs_path):
        dstpath = imgpath.replace("samplepose","boost")
        dstpath = os.path.join("..", dstpath)
        detection(
            img_name=imgpath, 
            model_pose=models['pose'], model_seg=models['seg'],
            out_name=dstpath
        )

if __name__ == "__main__":
    models = {
        'seg':YOLO('yolov8n-seg.pt'),
        'pose':YOLO('yolov8n-pose.pt')
    }
    

    main(models,walk_dir("samplepose"))

