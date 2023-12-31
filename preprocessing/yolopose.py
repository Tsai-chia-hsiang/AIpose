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
    results_seg = model_seg(img_name, verbose=False)#image name
    i=cv.imread(img_name)#get origin image
    #print(img_name)
    for r in results_seg:
        cls = r.boxes.cls.to(dtype=torch.int32)
        person_cls = torch.argwhere(cls == 0)[0]
        a_person = person_cls[0].item()
        m = (r.masks.data)[a_person]
        mask = torch.permute(m[None, :, :], (1, 2, 0)).numpy()
        mask_resize = cv.resize(mask, (i.shape[1], i.shape[0]))
        img = np.copy(i)
        img[:,:,0] = img[:,:,0]*mask_resize
        img[:,:,1] = img[:,:,1]*mask_resize
        img[:,:,2] = img[:,:,2]*mask_resize
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cnts, _ = cv.findContours(gray.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        clone = img.copy()*0
        cv.drawContours(clone, cnts, -1, (255, 255, 255), 2)
        results_pose = model_pose(img, verbose=False)
        #im_array = clone
        im_array = results_pose[0].plot(boxes=False, labels=False, img=clone)
        bbox = results_pose[0].boxes.xyxy.squeeze()
        bbox = bbox.to(dtype=int).tolist()
        im_array = im_array[bbox[1]:bbox[3], bbox[0]:bbox[2],:]
        cv.imwrite(out_name,im_array)

def main(models:dict, imgs_path:list):
    for imgpath in tqdm(imgs_path):
        dstpath = imgpath.replace("samplepose","sample_skeleton")
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
    
    
    """
    detection(
        img_name="samplepose\\Lake and ocean\\female\\2.jpg", 
        model_pose=models['pose'], model_seg=models['seg'],
        out_name="test.jpg"
    )
    """
    main(models,walk_dir("samplepose"))

