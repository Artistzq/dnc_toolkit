# %%
from typing import Any
import numpy as np

from finder import Finder
from ...utils.context_manager import Timer
from ...datasets.wrapper import Denormalize


class BlackOutputGetter:
    def __init__(self, model, lib="PyTorch") -> None:
        self.model = model
        self.lib = lib
    
    def __call__(self, x) -> Any:
        return self.get_output(x)
    
    def get_output(self, x):
        """返回x的黑盒输出，numpy数组
        Args:
            x (_type_): 输入样本
        """
        if self.lib == "PyTorch":
            import torch
            x: torch.Tensor = x
            x = x.unsqueeze(0)
            # print(x)
            # print(x.shape)
            # print(x.dtype)
            # print(x.device)
            # print(self.model)
            with torch.no_grad():
                output = self.model(x).to("cpu")
            return np.array(output)
        raise ValueError()

# %%
import random

import cv2
import numpy as np


def image_pixel_change(img):
    params = list(range(0, 10))

    random_params = random.sample(params, 1)

    # random change 1 - 5 pixels from 0 -255
    img_shape = img.shape
    img1d = np.ravel(img)
    arr = np.random.randint(0, len(img1d), random_params)
    for i in arr:
        img1d[i] = np.random.randint(0, 256)
    new_img = img1d.reshape(img_shape)
    return new_img


def image_noise(img):
    params = list(range(1, 4))
    random_params = random.sample(params, 1)
    if random_params == 1:
        return image_noise_gd(img)
    elif random_params == 2:
        return image_noise_rp(img)
    elif random_params == 3:
        return image_noise_ml(img)


def image_noise_gd(img):  # Gaussian-distributed additive noise.
    row, col, ch = img.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape([row, col, ch])
    noisy = img + gauss
    return noisy.astype(np.uint8)


def image_noise_rp(img):  # Replaces random pixels with 0 or 1.
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(img)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_salt))
              for i in img.shape]
    out[tuple(coords)] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i, int(num_pepper))
              for i in img.shape]
    out[tuple(coords)] = 0
    return out


def image_noise_ml(img):
    # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
    row, col, ch = img.shape
    mean = -0.5 + np.random.rand()
    var = 0.05 * np.random.rand()
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape([row, col, ch])
    # noisy = img + gauss
    #
    # gauss = np.random.randn(row, col, ch)
    # gauss = gauss.reshape([row, col, ch])
    noisy = img + img * gauss
    return noisy.astype(np.uint8)


def image_blur1(img):
    kernel = random.sample([1, 2, 3, 4, 5], 1)[0]
    # for opencv, image is HWC
    # if img.shape[0] > 3:
    result = cv2.blur(img, (kernel, kernel))
    if len(result.shape) == 2:
        result = result[..., np.newaxis]
    return result


def image_blur2(img):
    kernel = random.sample([1, 3, 5, 7], 1)[0]
    # if img.shape[0] > 3:
    result = cv2.GaussianBlur(img, (kernel, kernel), 0)
    if len(result.shape) == 2:
        result = result[..., np.newaxis]
    return result


def image_blur3(img):
    kernel = random.sample([1, 3, 5], 1)[0]
    # if img.shape[0] > 3:
    result = cv2.medianBlur(img, kernel)
    if len(result.shape) == 2:
        result = result[..., np.newaxis]
    return result



def get_img_mutations():
    # return [image_noise_gd, image_noise_rp, image_noise_ml, image_pixel_change]
    # return [image_noise_gd, image_noise_rp, image_pixel_change]
    return [image_noise_gd, image_noise_rp, image_noise_ml, image_pixel_change, image_blur1, image_blur2, image_blur3]


# %%
import numpy as np
import time


class MutationP:
    def __init__(self, m, total=0, delta_bigger_than_zero=0, epsilon=1e-7):
        self.mutation_method = m
        if hasattr(self.mutation_method, "__name__"):
            self.name = self.mutation_method.__name__
        else:
            self.name = time.perf_counter()
        self.total = total
        self.delta_bigger_than_zero = delta_bigger_than_zero
        self.epsilon = epsilon

    def mut(self, img):
        return np.clip(self.mutation_method(img), 0, 255)

    @property
    def score(self, epsilon=1e-7):
        # mylogger = Logger()
        rate = self.delta_bigger_than_zero / (self.total + epsilon)
        # mylogger.info("Name:{}, rate:{}".format(self.name, rate))
        return rate


class ProbabilityImgMutations:

    def __init__(self, ops, random_seed):
        self.p = 1 / len(ops)
        self.mutation_method = [MutationP(m) for m in ops]
        self.random = np.random.RandomState(random_seed)
        self.num_mutation_method = len(self.mutation_method)

    def add_mutation(self, m):
        if not isinstance(m, MutationP):
            m = MutationP(m)
        self.mutation_method.append(m)
        self.num_mutation_method = len(self.mutation_method)

    @property
    def mutators(self):
        mus = {}
        for mu in self.mutation_method:
            mus[mu.name] = mu
        return mus

    def select(self, mu1=None):
        if mu1 is None:
            # which means it's the first mutation
            return self.mutation_method[np.random.randint(0, self.num_mutation_method)]
        else:
            self.sort_mutators()
            k1 = self.index(mu1)
            k2 = -1
            prob = 0
            while self.random.rand() >= prob:
                k2 = self.random.randint(0, self.num_mutation_method)
                prob = (1 - self.p) ** (k2 - k1)
            mu2 = self.mutation_method[k2]
            return mu2

    def sort_mutators(self):
        import random
        random.shuffle(self.mutation_method)
        self.mutation_method.sort(key=lambda mutator: mutator.score, reverse=True)

    def index(self, mutator_name):
        for i, mu in enumerate(self.mutation_method):
            if mu.name == mutator_name:
                return i
        return -1


class RandomImgMutations(ProbabilityImgMutations):
    def select(self, mu1=None):
        idx = self.random.randint(low=0, high=self.num_mutation_method)
        return self.mutation_method[idx]


# %%
import pyflann
import numpy as np

# This class covered_states is used to record the covered states
# by the current testing input.

# Its current implementation is based on the tensorfFuzz
# https://github.com/brain-research/tensorfuzz/blob/master/lib/corpus.py

_BUFFER_SIZE = 5
_INIT_SIZE = 1

class CoveredStates(object):
    """Class holding the state of the update function."""

    def __init__(self, threshold=0.50, algorithm="kdtree"):
        """Inits the object.
        Args:
          threshold: Float distance at which coverage is considered new.
          algorithm: Algorithm used to get approximate neighbors.
        Returns:
          Initialized object.
        """
        self.flann = pyflann.FLANN()
        self.threshold = threshold
        self.algorithm = algorithm
        self.corpus_buffer = []
        self.lookup_array = []

        self.corpus = []

    def build_index_and_flush_buffer(self):
        """Builds the nearest neighbor index and flushes buffer of examples.
        This method first empties the buffer of examples that have not yet
        been added to the nearest neighbor index.
        Then it rebuilds that index using the contents of the whole corpus.
        Args:
          corpus_object: InputCorpus object.
        """
        self.corpus_buffer[:] = []
        self.lookup_array = np.vstack(
            self.corpus
        )
        self.flann.build_index(self.lookup_array, algorithm=self.algorithm)
        # tf.logging.info("Flushing buffer and building index.")

    def update_function(self, element):
        """Checks if coverage is new and updates corpus if so.
        The updater maintains both a corpus_buffer and a lookup_array.
        When the corpus_buffer reaches a certain size, we empty it out
        and rebuild the nearest neighbor index.
        Whenever we check for neighbors, we get exact neighbors from the
        buffer and approximate neighbors from the index.
        This stops us from building the index too frequently.
        FLANN supports incremental additions to the index, but they require
        periodic rebalancing anyway, and so far this method seems to be
        working OK.
        Args:
          corpus_object: InputCorpus object.
          element: CorpusElement object to maybe be added to the corpus.
        """
        if len(self.corpus) == 0:
            # print("waiting for element")
            self.corpus.append(element)
            self.build_index_and_flush_buffer()
            return True, 100
        else:
            _, approx_distances = self.flann.nn_index(
                element, 1, algorithm=self.algorithm
            )
            exact_distances = [
                np.sum(np.square(element - buffer_elt))
                for buffer_elt in self.corpus_buffer
            ]
            nearest_distance = min(exact_distances + approx_distances.tolist())
            if nearest_distance > self.threshold:
                # tf.logging.info(
                #     "corpus_size %s mutations_processed %s",
                #     len(corpus_object.corpus),
                #     corpus_object.mutations_processed,
                # )
                # tf.logging.info(
                #     "coverage: %s, metadata: %s",
                #     element.coverage,
                #     element.metadata,
                # )
                self.corpus.append(element)
                self.corpus_buffer.append(element)
                if len(self.corpus_buffer) >= _BUFFER_SIZE:
                    self.build_index_and_flush_buffer()
                return True, nearest_distance
            else:
                return False, nearest_distance


# %%
import numpy as np
from scipy.special import softmax
import torch

class CoverageBasedNN:

    def __init__(self) -> None:
        self.covered_states = CoveredStates()

    def is_exist(self, elem):
        existed, distance = self.covered_states.update_function(elem)


class DFlare(Finder):

    def __init__(self, model1: BlackOutputGetter, model2: BlackOutputGetter, timeout, delta, normalization, device) -> None:
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.timeout = timeout
        self.normalization = np.array(normalization)
        self.mean = self.normalization[0].reshape((-1, 1, 1))
        self.std = self.normalization[1].reshape((-1, 1, 1))
        self.delta = delta
        self.device = device

    def tensor2img(self, x):
        x = x.to("cpu")
        # x 标准化后的图像，有负数
        original_image = (x * self.std) + self.mean
        return np.array(original_image * 255, dtype=np.uint8).clip(0, 255)
    
    def img2tensor(self, img):
        img = np.array(img, dtype=np.float32)
        img /= 255
        x = torch.from_numpy((img - self.mean) / self.std).to(self.device).float()
        return x
        
    def H(self, img):
        logits_f = self.model1(self.img2tensor(img))
        logits_g = self.model2(self.img2tensor(img))
        return self.fitness(logits_f, logits_g)
    
    def fitness(self, logits_f, logits_g):
        # 先计算差值
        probs_f = softmax(logits_f, axis=-1)[0]
        probs_g = softmax(logits_g, axis=-1)[0]
        
        label_f = np.argmax(probs_f)
        label_g = np.argmax(probs_g)
        diff = np.abs(probs_f[label_f] - probs_g[label_g])
        
        # 再通过knn计算o
        existed = self.nn.is_exist(np.hstack([logits_f, logits_g]))
        o = 1 if existed else 0
        return diff / self.delta + o
    
    def __call__(self, xs):
        # xs是标准化后的图片，（-xx, xx)，图片变异算子要求图片是0-255
        # 计算过程中，全程转换为图片（0-255）
        
        self.pool = ProbabilityImgMutations(get_img_mutations(), 42)
        self.nn = CoverageBasedNN()
        
        xs = self.tensor2img(xs)
        best_img = np.copy(xs)
        last_mutation_operator = None
        
        best_fitness_value = self.H(best_img)
        with Timer(time_unit="s", verbose=False) as timer:
            count = 0
            while timer.get_elapsed_time() < self.timeout:
                # 选择op
                op = self.pool.select(last_mutation_operator)
                op.total += 1
                # 突变
                # print(best_img.shape)
                # print(best_img.dtype)
                new_img = op.mut(np.copy(best_img))
                count += 1
                # 计算突变后的输出
                # print(1)
                logits_f = self.model1(self.img2tensor(new_img))
                # print(2)
                logits_g = self.model2(self.img2tensor(new_img))
                # print(3)
                fitness_value = self.fitness(logits_f, logits_g)
                # print(4)
                
                if np.argmax(logits_f) != np.argmax(logits_g):
                    best_img = new_img
                    # op.delta_bigger_than_zero += 1
                    # print("生成成功。突变{}次".format(count))
                    return self.img2tensor(best_img)
                
                # 出现了更好的图片，就更新
                if fitness_value >= best_fitness_value:
                    best_fitness_value = fitness_value
                    best_img = new_img
                    op.delta_bigger_than_zero += 1
                    last_mutation_operator = op
        print("生成失败败败败败。")
        return self.img2tensor(best_img)

if __name__ == "__main__":

    # %% [markdown]
    # # 测试

    # %%
    import math
    import torch
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    # 计算PSNR和SSIM
    def calculate_psnr_ssim(tensor1, tensor2):
        psnr_value = peak_signal_noise_ratio(tensor1, tensor2, data_range=1.0)
        ssim_value = structural_similarity(tensor1, tensor2, channel_axis=2)
        return psnr_value, ssim_value

    # 计算每对图像的PSNR和SSIM
    def calculate_image_metrics(image_list1, image_list2):
        psnr_list = []
        ssim_list = []

        for img1, img2 in zip(image_list1, image_list2):
            psnr, ssim = calculate_psnr_ssim(img1, img2)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        return psnr_list, ssim_list


    from toolkit.commons import *
    import math

    # setting_name = "TinyImageNet"
    # args = Args(setting_name)

    # # 读取数据集
    # dataset = get_dataset(args.dataset, batch_size=args.b)

    # # 读取模型
    # large = get_network(args.net, dataset.num_classes, weight=args.weight)
    # pmodel = torch.load(args.prune_weight)
    # # qmodel = TorchQuantizer.load_model(large, args.quant_weight)
    # # kd1 = get_network(args.kd1, dataset.num_classes, weight=args.kd1_weight)
    # # kd2 = get_network(args.kd2, dataset.num_classes, weight=args.kd2_weight)

    # smalls = {
    #     "P": pmodel,
    #     # "KD1": kd1,
    #     # "KD2": kd2,
    #     # "Q": qmodel
    # }

    # for i, (name, model) in enumerate(smalls.items()):
    #     device = 'cpu' if name == "Q" else "cuda"
    #     white_model = model
    #     black_model = large
    #     if name == "Q":
    #         white_model, black_model = black_model, white_model
    #     white_model.to(device)
    #     black_model.to(device)

    #     same_tensor = SameFinder.find_images(large, model, 10, dataset.test_loader, device=device, agreement=True)

    #     dflare = DFlare(
    #         BlackOutputGetter(model),
    #         BlackOutputGetter(large),
    #         timeout=20, 
    #         delta=1e-3,
    #         normalization=MEAN_STDs[setting_name],
    #         device=device
    #     )

    #     with Timer(task="DFlare") as timer:
    #         imgs = [dflare(x) for x in same_tensor[0]]
    #     imgs = torch.stack(imgs)
        
    #     dr = ModelMetric(wrapper.to_loader((imgs, same_tensor[1]))).disagree_rate(large, model)
        
    #     # 调用函数计算PSNR和SSIM
        
    #     W = wrapper.get_plot_wrapper(MENAs[setting_name])
    #     psnr_values, ssim_values = calculate_image_metrics(W(same_tensor[0]), W(imgs))

    #     print(setting_name)
    #     print(name)
    #     filtered_psnr_values = [num for num in psnr_values if not math.isinf(num)]
    #     print("PSNR values:", sum(filtered_psnr_values) / len(filtered_psnr_values))
    #     print("SSIM values:", sum(ssim_values) / len(ssim_values))
    #     print("SR: {}%".format(round(dr * 100, 2)))
    #     # print("Time: {}s".format(times[i]))
        

    # %%
    def show_result(datasets, image_nums):
        setting_name = datasets
        args = Args(setting_name)

        # 读取数据集
        dataset = get_dataset(args.dataset, batch_size=args.b)

        # 读取模型
        large = get_network(args.net, dataset.num_classes, weight=args.weight)
        pmodel = torch.load(args.prune_weight)
        qmodel = TorchQuantizer.load_model(large, args.quant_weight)
        kd1 = get_network(args.kd1, dataset.num_classes, weight=args.kd1_weight)
        kd2 = get_network(args.kd2, dataset.num_classes, weight=args.kd2_weight)

        smalls = {
            # "P": pmodel,
            # "KD1": kd1,
            # "KD2": kd2,
            "Q": qmodel
        }

        for i, (name, model) in enumerate(smalls.items()):
            device = 'cpu' if name == "Q" else "cuda"
            # white_model = model
            # black_model = large
            # if name == "Q":
            #     white_model, black_model = black_model, white_model
            # white_model = white_model.to(device)
            # black_model = black_model.to(device)
            
            large = large.to(device)
            model = model.to(device)
            
            

            same_tensor = SameFinder.find_images(large, model, image_nums, dataset.test_loader, device=device, agreement=True)

            
            dflare = DFlare(
                BlackOutputGetter(model),
                BlackOutputGetter(large),
                timeout=20, 
                delta=1e-3,
                normalization=MEAN_STDs[setting_name],
                device=device
            )

            with Timer(task=name) as timer:
                imgs = [dflare(x) for x in same_tensor[0]]
            imgs = torch.stack(imgs)
            dr = ModelMetric(wrapper.to_loader((imgs, same_tensor[1])), use_gpu=False).disagree_rate(large, model)
            # 调用函数计算PSNR和SSIM
            
            W = wrapper.get_plot_wrapper(MENAs[setting_name])
            psnr_values, ssim_values = calculate_image_metrics(W(same_tensor[0]), W(imgs))

            print(setting_name)
            print(name)
            filtered_psnr_values = [num for num in psnr_values if not math.isinf(num)]
            print("PSNR values:", sum(filtered_psnr_values) / len(filtered_psnr_values))
            print("SSIM values:", sum(ssim_values) / len(ssim_values))
            print("SR: {}%".format(round(dr * 100, 2)))
        return same_tensor, W, imgs

    # %%
    same_tensor, W, imgs = show_result("CIFAR10", 10)

    # %%
    print(same_tensor[0].shape)
    print(imgs.shape)

    # %%
    import matplotlib as mpl
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np


    def imshow(image_group, y_labels=None):
        b = 2
        fig, axes = plt.subplots(len(image_group), len(image_group[0]), figsize=(len(image_group[0])*b, len(image_group)*b))

        # Add y_labels to the leftmost column
        if y_labels:
            for i, label in enumerate(y_labels):
                axes[i, 0].set_ylabel(label, rotation=90, ha='center', va='center')
                
                # Remove y-axis ticks
                axes[i, 0].set_yticks([])

        for i, images in enumerate(image_group):
            for idx, image in enumerate(images):
                ax = axes[i][idx]
                ax.imshow(image)

                # Remove x-axis ticks
                ax.set_xticks([])
                ax.set_yticks([])

        plt.show()

    # %%
    imshow([W(same_tensor[0][:10]), W(imgs[:10])], y_labels=['Origin', 'DF'])

    # %%
    torch.save(imgs, "imgs.pt")

    # %%



