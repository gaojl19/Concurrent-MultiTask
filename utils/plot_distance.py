import json
from sqlite3 import DataError
import json
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio, os

HARD_MT10_CLS_DICT = {
    'button-press-v1': 1,
    'button-press-wall-v1': 1,
    'push-v1': 1,
    'pick-place-v1': 1,
    'drawer-open-v1': 1,
    'peg-insert-side-v1': 1,
    'push-wall-v1': 1,
    'pick-place-wall-v1': 1,
    'coffee-pull-v1': 1,
    'stick-push-v1': 1
}

def calculate_distance(w1_dict, w2_dict):
    from scipy.spatial.distance import pdist
    total_step = min(len(w1_dict), len(w2_dict))
    distance_list = []
    for i in range(total_step):
        x = np.array(w1_dict[str(i)])
        y = np.array(w2_dict[str(i)])
        X=np.vstack([x,y])
        d=pdist(X,'cityblock') / x.shape[0]
        print(d)
        
        distance_list.append(d)

    return distance_list

        
        
def plot_distance(path, curve, task_tag):
    
    cnt = 0
    print(path)
        
    for distance in curve:
        d = np.array(distance).reshape(-1)
        ax = sns.barplot(x=task_tag, y=d, palette="BuPu") 
        plt.xticks(rotation=15,fontsize=8)
        plt.figure(figsize=(8, 8)),
        ax.set(ylim=(0, 1))
        fig = ax.get_figure()
        fig.savefig(path + str(cnt) + ".png")
        plt.close()
        cnt += 1


def generate_gif(path, directory, task_name):
    images = []
    print(directory)
    filenames=sorted((fn for fn in os.listdir(directory) if fn.endswith('.png')))
    for filename in filenames:
        images.append(imageio.imread(directory + "/" + filename))
    imageio.mimsave(path + task_name+'_distance.gif', images, duration=1)

        

if __name__ == "__main__":
    
    task_weight = {}
    path="./test/mt10_hard_2_2_256_fixed/"
    
    for task in HARD_MT10_CLS_DICT.keys():
        json_path = path + task + "_weight.json"
        with open(json_path, "r") as f:
            weights = json.load(f)
        task_weight[task] = weights
        f.close()
        
    for task1 in HARD_MT10_CLS_DICT.keys():
        distance_list = []
        task_tag = []
        for task2 in HARD_MT10_CLS_DICT.keys():
            if (task1 == task2):
                continue
            else:
                d = calculate_distance(task_weight[task1], task_weight[task2])
                d = calculate_distance(task_weight[task2], task_weight[task1])
                distance_list.append(d)
                task_tag.append(task2)
                # if task1 == "button-press-v1" and task2 == "button-press-wall-v1":
                #     print(d)
                # elif task2 == "button-press-v1" and task1 == "button-press-wall-v1":
                #     print(d)
        
        curve = []
        for i in range(len(distance_list[0])):
            sub_curve = []
            for d in distance_list:
                # print(d[0])
                sub_curve.append(d[i])
            curve.append(sub_curve)
            
        path_new = path + task1 + "_distance/"
        
        import shutil  
        shutil.rmtree(path_new)  
        os.mkdir(path_new)
        if not os.path.isdir(path_new):
            os.makedirs(path_new)
            
        plot_distance(path_new, curve, task_tag)       
        generate_gif(path, path_new, task1)
        exit(0)