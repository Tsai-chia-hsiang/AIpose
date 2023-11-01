import tkinter as tk
import os
import cv2
import gc
from PIL import Image, ImageTk
from main import walk_dir, combine_pose_to_scene, predict

class app():

    def __init__(self, demo_scene:list, gender='female') -> None:

        self.__root = tk.Tk()
        self.__user_gender = gender
        self.__root.title('AI Pose cconcept')
        self.__root.geometry('800x800')
        self.__sample_scene = {
            'path':demo_scene.copy(),
            'image':[],
            'button':[], 
            'coordinate':[[20,20],[20,280],[20,540]],
            'label':[None, None, None]
        }
        self.__demo_scene_with_pose = None
        self.__img = None
        self. __read_and_set_photo_bnts()
        

    def __read_and_set_photo_bnts(self):

        for idx, i in enumerate(self.__sample_scene['path']):
            j = Image.open(i)
            self.__sample_scene['image'].append(
                [j, ImageTk.PhotoImage(j.resize((150,250)))]
            )
            self.__sample_scene['button'].append(
                tk.Button(
                    self.__root, text=i, 
                    image=self.__sample_scene['image'][idx][1],
                    command=lambda c=idx:self.__predict(c)
                )
            )
            self.__sample_scene['button'][idx].place(
                x = self.__sample_scene['coordinate'][idx][0],
                y = self.__sample_scene['coordinate'][idx][1]
            )

    def __predict(self, i:int):
    
        if self.__sample_scene['label'] is not None:
            p = predict(self.__sample_scene['image'][i][0])
            cluster_dir = os.path.join(
                "sample_skeleton", p['topic'][1], self.__user_gender
            )
            self.__sample_scene['label'][i] = (p['label'], cluster_dir)
    
        print(self.__sample_scene['label'])
        self.__photoshopped(i)
    
    def __photoshopped(self, i:int):
        selection = walk_dir(self.__sample_scene['label'][i][1])
        print(self.__sample_scene['path'][i])
        self.__img = Image.fromarray(
            cv2.cvtColor(
                combine_pose_to_scene(self.__sample_scene['path'][i], selection[0]),
                cv2.COLOR_BGR2RGB
            )
        )

        if self.__demo_scene_with_pose is not None:
            del self.__demo_scene_with_pose
            gc.collect()
        self.__img = ImageTk.PhotoImage(self.__img)       
        self.__demo_scene_with_pose = tk.Label(self.__root, image=self.__img)
        self.__demo_scene_with_pose.place(x=150, y=20)


    def __call__(self) -> None:
        self.__root.mainloop()

if __name__ == "__main__":
    print("demo")
    a = app(demo_scene=walk_dir(os.path.join("scenes")))
    a()