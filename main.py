from places365 import run
import sys
import os
import cv2
import numpy as np
import os.path as osp

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

def combine_pose_to_scene(scene:os.PathLike|np.ndarray, pose:os.PathLike|np.ndarray)->np.ndarray:
    
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


def run_smaple(sampleroot:os.PathLike): 

    resultroot = osp.join(sampleroot, "result")
    if not osp.exists(resultroot):
        os.mkdir(resultroot)

    for smaplesecen in walk_dir(sampleroot):
        p = predict(smaplesecen)
        cluster_dir = osp.join("sample_skeleton", p['topic'][1])
        suggestion_poses = walk_dir(cluster_dir)
        d,f = osp.split(smaplesecen)
        add_pose_img_path = os.path.join(resultroot, f)
        combine = combine_pose_to_scene(smaplesecen, suggestion_poses[0])
        cv2.imwrite(add_pose_img_path, combine)

if __name__ == "__main__":
    run_smaple(sys.argv[1])