# %%

import matplotlib as mpl
import numpy as np
 
# data = np.array(data).T
# plt.bar()

# plt.savefig("test.png", bbox_inches='tight', transparent=False)


import matplotlib
import matplotlib.pyplot as plt
import numpy as np



def im_show(X):
    if len(X.shape) == 4:
        X = X[0];
    
    X = X.cpu().numpy()
    X.swapaxes(0, 2)
    X.swapaxes(0, 1)
    
    






label = ["class"+str(i) for i in range(10)]

def create_multi_bars(labels, datas, title, tick_step=1, group_gap=0.2, bar_gap=0):
    '''
    labels : x轴坐标标签序列
    datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致
    tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
    group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
    bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    '''
    # x为每组柱子x轴的基准位置
    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    # 绘制柱子
    for index, y in enumerate(datas):
        plt.bar(x + index*bar_span, y, bar_width)
    plt.ylabel('Acc Difference')
    plt.title(title)
    # ticks为新x轴刻度标签位置，即每组柱子x轴上的中心位置
    ticks = x + (group_width - bar_span) / 2
    plt.xticks(ticks, labels)
    # plt.ylim(80, 100)
    # plt.show()
    plt.savefig("{}.png".format(title), bbox_inches='tight', transparent=False)


# plt.savefig("test.png", bbox_inches='tight', transparent=False)

# %%
import os
import matplotlib.pyplot as plt
import json
import numpy as np

results = [item for item in os.listdir("./results") if "json" in item]

for result in results:
    path = os.path.join("./results", result)
    name = "_".join(result.split("_")[:-1])
    model_type = name.split("_")[0]
    # 一个模型的数据
    print(model_type)
    records = json.load(open(path, "r"))
    # 取origin model 和 前4次ban
    records = records[:5]
    all_acc = [round(record["acc"]* 100, 2) for record in records]
    all_ece_10 = [round(record["ece_10"] * 100, 2) for record in records]
    all_nfr = [record.get("nfr", 0) for record in records]
    
    # print("acc ", all_acc)
    # print("ece ", all_ece_10)
    # print("nfr ", all_nfr)
    
    data = []
    all_class_acc = [record["class_acc"] for record in records]
    # for i in range(len(all_class_acc)):
    #     data.append(np.array(all_class_acc[i]) - np.array(all_class_acc[0]).tolist())
    data = all_class_acc
    labels = ["class " + str(i) for i in range(10)]
    # print(all_class_acc)
    # print(data)
    create_multi_bars(labels, data, model_type, group_gap=0.5)
    # print(all_class_acc)
    # fig, axes=plt.subplots(1, 3, figsize=(12, 4))
    # axes[0].plot(range(len(data)), accs)
    # axes[0].set_title("acc")
    # axes[1].plot(range(len(data)), ece_10s)
    # axes[1].set_title("ece_10")
    # axes[2].plot(range(len(data)), ece_15s)
    # axes[2].set_title("ece_15")
    # fig.suptitle(result)
    # fig.tight_layout()
    
#     fig.savefig(args.path + "_pngs/{}_".format(result) + ".png", bbox_inches='tight', transparent=False)

# with open("./results/resnet18_15ep/results.txt") as f:
#     data = f.readlines()

# data = data[:4]
# data = [item.split(":")[-1] for item in data]
# data = [[float(v) for v in item.split(" ")] for item in data]


# %%



