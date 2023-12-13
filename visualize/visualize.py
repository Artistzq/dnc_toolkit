import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# def imshow(image_group, titles=None):
#     b = 2
#     fig, axes = plt.subplots(len(image_group), len(image_group[0]), figsize=(len(image_group[0])*b, len(image_group)*b))
#     for i, images in enumerate(image_group):
#         for idx, image in enumerate(images):
#             axes[i][idx].imshow(image)
#             axes[i][idx].set_axis_off()
            
#         if titles:
#             axes[i][idx].set_suptitle(titles[i])

#     fig.show()

def imshow(image_group, titles=None):
    b = 2
    fig, axes = plt.subplots(len(image_group), len(image_group[0]), figsize=(len(image_group[0])*b, len(image_group)*b))
    for i, images in enumerate(image_group):
        for idx, image in enumerate(images):
            ax = axes[i][idx]
            ax.imshow(image)
            ax.set_axis_off()

            # Add titles to the first subplot in each row
            if idx == 0 and titles is not None:
                ax.set_title(titles[i], rotation='vertical', ha='right')

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


def heatmap():
    plt.imshow(cmap="inferno")
    
    

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
