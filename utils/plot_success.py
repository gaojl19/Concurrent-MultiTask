from json.tool import main
import json
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter

def plot_success_curve(path1, path2, label1, label2, tag, plot_prefix):
    
    json_path = path1 + "_success.json"
    with open(json_path, "r") as f:
        json_file = json.load(f)
        curve1 = list(json_file.values())
    
    json_path = path2 + "_success.json"
    with open(json_path, "r") as f:
        json_file = json.load(f)
        curve2 = list(json_file.values())
    
    iteration = range(1, max(len(curve1), len(curve2))+1)
    if len(curve1) > len(curve2):
        for _ in range(len(curve1)-len(curve2)):
            curve2.append(curve2[-1])
    else:
        for _ in range(len(curve2)-len(curve1)):
            curve1.append(curve1[-1])
            
            
    # smooth
    curve1 = savgol_filter(curve1, 11, 3)
    curve2 = savgol_filter(curve2, 11, 3)
    
    df = {  label1: curve1,
            label2: curve2 }
    
    sns.set("paper")
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("bright", 2)
    
    data = pd.DataFrame(data=df, index=iteration)
    ax=sns.lineplot(data=data, palette=palette)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(tag + " success")
    ax.set_title(tag + " Mean Success Curve")
    ax.set(ylim=(-0.1, 1.1))
    
    fig = ax.get_figure()
    fig.savefig(plot_prefix + tag + "_success_curve.png")
    
    fig.clf()


def plot_all_curve(paths, labels, tag, plot_prefix):
    
    curve = []
    max_len = 0
    for path in paths: 
        json_path = path + "_success.json"
        with open(json_path, "r") as f:
            json_file = json.load(f)
            origin_success = list(json_file.values())
            
            # smooth
            new_curve = savgol_filter(origin_success, 101, 3)
            curve.append(new_curve)
            max_len = len(new_curve) if len(new_curve) > max_len else max_len
        
            
    iteration = range(1, max_len+1)
    print(iteration)
    assert len(labels) == len(curve)
    df = {}
    
    for i in range(len(curve)):
        c = curve[i].tolist()
        padding = max_len - len(c) if max_len >len(c) else 0
        for _ in range(padding):
            c.append(c[-1])
        df[labels[i]] = c
    
    sns.set("paper")
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("bright", len(curve))
    
    data = pd.DataFrame(data=df, index=iteration)
    ax=sns.lineplot(data=data, palette=palette)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(tag + " success")
    ax.set_title(tag + " Mean Success Curve")
    ax.set(ylim=(-0.1, 1.1))
    
    fig = ax.get_figure()
    fig.savefig(plot_prefix + tag + "_success_curve.png")
    
    fig.clf()
    
    
if __name__ == "__main__":
    # path1 = "./fig/mt10_hard_8_8_32_fixed/"
    # path2 = "./fig/mt10_hard_8_8_32_bn/"
    # label1 = "Soft-Module baseline"
    # label2 = "Soft-Module with batch norm"
    # task_name = "SoftModule_2_2_256_Batch_Norm"
    # plot_prefix = "./fig/MT10_Hard_Result/"
    
    # plot_success_curve(path1=path1, path2=path2, label1=label1, label2=label2, tag=task_name, plot_prefix=plot_prefix)
    
    keyword = "_dense"
    task_group = "diverse"
    path_prefix = "./fig/MT10 Diverse Baseline/"
    path = [ path_prefix + "mt10_" + task_group + "_mh_fixed/", 
             path_prefix + "mt10_" + task_group + "_baseline_fixed/",
             path_prefix + "mt10_" + task_group + "_2_2_256" + keyword + "_fixed/",
             path_prefix + "mt10_" + task_group + "_2_10_64"+ keyword + "_fixed/",
             path_prefix + "mt10_" + task_group + "_4_4_64"+ keyword + "_fixed/",
             path_prefix + "mt10_" + task_group + "_4_4_256"+ keyword + "_fixed/",
             path_prefix + "mt10_" + task_group + "_8_2_128"+ keyword + "_fixed/",
             path_prefix + "mt10_" + task_group + "_8_8_32"+ keyword + "_fixed/"]
    label = [ "multi-head", "MLP", "2/2/256", "2/10/64", "4_4_64", "4_4_256", "8_2_128", "8_8_32"]
    tag = "MT10 Diverse"
    plot_prefix = "./fig/MT10_Diverse_Result/"
    plot_all_curve(path, label, tag, plot_prefix)
    