from places365 import run
import sys
import os
import os.path as osp
from PIL import Image

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

def main(cls=False, img = None):
    img = sys.argv[1] if cls else img 
    p = predict(img)
    print(p)
    cluster_dir = osp.join("sample_skeleton", p['topic'][1])
    suggestion_poses = walk_dir(cluster_dir)
    print(suggestion_poses)

if __name__ == "__main__":
    main(cls=True)