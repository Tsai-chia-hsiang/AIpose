from places365 import run
import os
import cv2
import gc
import numpy as np
import tkinter as tk
import numpy as np
import cv2  
from PIL import Image, ImageTk

place365rn50 = run.Place365_Pretrianed_ResNet50(
    modelat=run.p365dir, classespath=run.class_fname,
    label_clustering_file=run.label_clustering_file
)

def predict(scene)->dict:
    r = place365rn50.predict(scene, topk=1)
    return r[0]

def walk_dir(root:os.PathLike)->list:
    u = []
    for r, dirs, files in os.walk(root, topdown=False):
        for f in files:
            u.append(os.path.join(r, f))             
    return u


class app():

    def __init__(self, demo_scene:list, gender='female') -> None:

        self.__root = tk.Tk()
        self.__phone_window_shape = (313,500)
        self.__user_gender = gender
        self.__root.title('AI Pose cconcept')
        self.__root.geometry('800x800')
        self.__sample_scene = {
            'path':demo_scene.copy(),
            'image':[],
            'np_demo_image':[],
            'button':[], 
            'coordinate':[[20,20],[20,280],[20,540]],
            'label':[None, None, None]
        }
        self.__pose_selection =None
        self.__pose = {
            'poseimg':[],
            'bnt':[]
        }
        self.__demo_scene = None
        self.__img = None
        self.__read_and_set_photo_bnts()
        self.__phone_scene = {'phone':[],'srcimage':[]}
        self.__next_pose_set_bnt = None
        self.__previous_pose_set_bnt = None
        self.__phone_background()
        self.__poseid_base = 0
        self.__select_scene = None
        
    def __phone_background(self):

        img = Image.open('phone.jpg')
        self.__phone_scene['srcimage'].append(ImageTk.PhotoImage(img.resize((354,719))))
        self.__phone_scene['phone'].append(tk.Label(self.__root, image = self.__phone_scene['srcimage'][0]))
        self.__phone_scene['phone'][0].place(x=300, y=10)

    def __read_and_set_photo_bnts(self):

        for idx, i in enumerate(self.__sample_scene['path']):
            j = Image.open(i)
            
            self.__sample_scene['np_demo_image'].append(
                cv2.resize(np.array(j), self.__phone_window_shape)
            )

            self.__sample_scene['image'].append([j, ImageTk.PhotoImage(j.resize((150,250)))])
            self.__sample_scene['button'].append(
                tk.Button(
                    self.__root, text=i, 
                    image=self.__sample_scene['image'][idx][1],
                    command=lambda c=idx:self.__show_scene(c)
                )
            )
            self.__sample_scene['button'][idx].place(
                x = self.__sample_scene['coordinate'][idx][0],
                y = self.__sample_scene['coordinate'][idx][1]
            )

    def __del_scene_label(self):
        if self.__demo_scene is not None:
            self.__demo_scene.destroy()

    def __del_pose_bnt(self):
    
        if len(self.__pose['bnt']) == 0:
            return
    
        for i in range(len(self.__pose['bnt'])):
            self.__pose['bnt'][i].destroy()

        self.__pose['poseimg'].clear()
        self.__pose['bnt'].clear()
        gc.collect()
    
    def __make_pose_bnt(self):
        pose_pathes = self.__pose_selection[self.__poseid_base:self.__poseid_base+3]
        for idx, p in enumerate(pose_pathes):
            i = Image.open(p)
            i = ImageTk.PhotoImage(i.resize((60,100)))
            self.__pose['poseimg'].append(i)
            self.__pose['bnt'].append(
                tk.Button(
                    self.__root,
                    image=self.__pose['poseimg'][idx], 
                    command=lambda c=idx:self.__photoshop(self.__poseid_base+c)
                )
            )
    
    def __place_pose_bar(self):
        offset = 0
        for idx in range(len(self.__pose['bnt'])):
            self.__pose['bnt'][idx].place(x = 355+offset, y = 120)
            self.__pose['bnt'][idx].lift(self.__demo_scene)
            offset += 85
    
    def __show_scene(self, i:int):
        self.__select_scene = i
        self.__del_scene_label()
        self.__del_pose_bnt()
        self.__poseid_base = 0
        self.__predict(i)
        self.__make_pose_bnt()
        self.__img = ImageTk.PhotoImage(Image.fromarray(self.__sample_scene['np_demo_image'][i]))    
        self.__demo_scene = tk.Label(self.__root, image=self.__img)
        self.__demo_scene.place(x=320, y=114)
        self.__place_pose_bar()
        
    def __predict(self, i:int):
    
        if self.__sample_scene['label'][i] is None:
            p = predict(self.__sample_scene['image'][i][0])
            cluster_dir = os.path.join("sample_skeleton", p['topic'][1], self.__user_gender)
            self.__sample_scene['label'][i] = (p['label'], cluster_dir)
            # print("predict scene")
        # print("prepare pose")
        self.__pose_selection = walk_dir(self.__sample_scene['label'][i][1]) 
    
    def __photoshop(self, poseid:int):

        self.__del_scene_label()
        
        self.__img = self.combine_pose_to_scene(
            self.__sample_scene['np_demo_image'][self.__select_scene].copy(), 
            cv2.cvtColor( 
                cv2.imread(self.__pose_selection[poseid]), 
                cv2.COLOR_BGR2RGB
            )
        )
        self.__img = ImageTk.PhotoImage(Image.fromarray(self.__img))    
        self.__demo_scene = tk.Label(self.__root, image=self.__img)
        self.__demo_scene.place(x=320, y=114)
        self.__place_pose_bar()

    def combine_pose_to_scene(self, scene:os.PathLike|np.ndarray, pose:os.PathLike|np.ndarray)->np.ndarray:
        
        def resize_pose_shape(poseshape:tuple, sceneshape:tuple)->tuple:
            f = 1.6 if np.argmax(sceneshape) == 0 else 1.0 
            ratio = poseshape[1]/poseshape[0]
            resize_aix_ratio = sceneshape[0]/f
            resize = [int(resize_aix_ratio),int(resize_aix_ratio*ratio)]
            #print(resize)
            if resize[1] > sceneshape[1]:
                resize[1] = int(sceneshape[1]/f)
            resize.reverse()
            return tuple(resize)
        
        s = cv2.imread(scene) if isinstance(scene, str) else scene
        spose = cv2.imread(pose) if isinstance(pose, str) else pose
        sshape = s.shape
        resize = resize_pose_shape(spose.shape, sshape)
        spose = cv2.resize(spose,resize)
        spose = spose.astype(np.float32)

        # print(sshape)
        # print(spose.shape)

        w_left = sshape[1]//2 - spose.shape[1]//2
        # print(w_left)
        m = (spose <= 85.0).astype(np.float32)
        posepart =s[s.shape[0]-10-spose.shape[0]:s.shape[0]-10, w_left:w_left+spose.shape[1],:] 
        s[s.shape[0]-10-spose.shape[0]:s.shape[0]-10, w_left:w_left+spose.shape[1],:] = posepart*m + spose
        return np.clip(0, 255, s).astype(np.uint8)

    def __call__(self) -> None:
        self.__root.mainloop()

if __name__ == "__main__":
    a = app(demo_scene=walk_dir(os.path.join("scenes")))
    a()