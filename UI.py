import tkinter as tk
from typing import Any
import os
from PIL import Image, ImageTk
from main import walk_dir

class app():
    def __init__(self, demo_scene:list) -> None:
        self.root = tk.Tk()
        self.root.title('AI Pose cconcept')
        self.root.geometry('1200x800')
        self.__sample_scene = {
            'image':[],
            'button':[], 
            'coordinate':[[20,20],[20,300],[20,530]]
        }
        self. __read_and_set_photo_bnts(demo_scene=demo_scene)

    def __read_and_set_photo_bnts(self, demo_scene:list):
        for idx, i in enumerate(demo_scene):
            j = Image.open(i)
            self.__sample_scene['image'].append([j, ImageTk.PhotoImage(j.resize((250,250)))])
            self.__sample_scene['button'].append(
                tk.Button(self.root, image=self.__sample_scene['image'][idx][1])
            )
            self.__sample_scene['button'][idx].place(
                x = self.__sample_scene['coordinate'][idx][0],
                y = self.__sample_scene['coordinate'][idx][1]
            )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.root.mainloop()

if __name__ == "__main__":
    print("demo")
    a = app(demo_scene=walk_dir(os.path.join("scenes")))
    a()