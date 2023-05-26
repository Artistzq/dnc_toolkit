import torch
import time
import random
import numpy as np
from torchvision.transforms import transforms
from .show import Denormalize

class DiffChaser:
    def __init__(self, 
                 model1, 
                 model2, 
                 iteration, 
                 timeout, 
                 population_size, 
                 normalization,
                 cross_rate=0.5,
                 mutate_rate=0.01,
                 distance_restrict=8/255,
                 verbose=False) -> None:
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.iteration = iteration
        self.timeout = timeout
        self.population_size = population_size
        self.normalizer = transforms.Normalize(mean=normalization[0], std=normalization[1])
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.pixel_limit = int(distance_restrict * 255)
        self.verbose = verbose
        
    def init_population(self, x):
        population = []
        for i in range(self.population_size):
            population.append(self.mutate(x))
        return population

    def fitness_function(self, x, model, ref, option="index"):
        x = x.unsqueeze(0)
        logits = model(x)[0]
        indicies = torch.argsort(-logits, dim=-1)
        
        if option == "index":
            return logits[indicies[0]] - logits[indicies[ref]]
        elif option == "target":
            return logits[indicies[0]] - logits[ref]
        else:
            return -1

    @DeprecationWarning
    def basic_fitness(self, x):
        return self.k_uncertainty_fitness(x, 1)
    
    @DeprecationWarning
    def k_uncertainty_fitness(self, x, k):
        assert k > 0, "k start from 1."
        logits = self.model1(x)
        indicies = torch.argsort(logits, dim=-1)
        return logits[indicies[0]] - logits[indicies[k]]
    
    @DeprecationWarning
    def t_target_fitness(self, x, t):
        logits = self.model1(x)
        indicies = torch.argsort(logits, dim=-1)
        return logits[indicies[0]] - logits[t]

    def crossover(self, x1, x2) -> torch.Tensor:
        """x1与x2以self.cross_rate的概率交叉
        """
        rand = torch.rand(x1.shape)
        lower_index = torch.where(rand < self.cross_rate)
        greater_index = torch.where(rand >= self.cross_rate)
        
        new_x1 = torch.zeros_like(x1)
        new_x1[lower_index] = x1[lower_index]
        new_x1[greater_index] = x2[greater_index]
        
        new_x2 = torch.zeros_like(x2)
        new_x2[lower_index] = x2[lower_index]
        new_x2[greater_index] = x1[greater_index]
        
        return new_x1, new_x2
    
    def in_distance_limit(self, x, ref, p, limit):
        sub = torch.abs(x - ref)
        if p == "infinity":
            max_val = torch.max(sub).item()
            if (max_val < limit):
                return True
        return False
        
    def mutate(self, x):
        mean = self.normalizer.mean
        std = self.normalizer.std
        x = Denormalize(mean, std)(x)
        
        # 计算最大限制
        limit = self.pixel_limit / 255
        for i in range(3):
            # 突变图像
            prob = torch.rand_like(x)
            # x+突变
            mutation = torch.rand_like(x) * (self.pixel_limit / 255)
            mutated = x + mutation
            # clip图片后个体
            mutated = torch.clip(mutated, 0, 1)
            # 另一部分还原为x
            index = torch.where(prob > self.mutate_rate)
            mutated[index] = x[index]
            
            if (self.in_distance_limit(x, mutated, "infinity", limit)):
                return self.normalizer(mutated)
        return self.normalizer(x)
    
    def agree(self, x) -> bool:
        x = x.unsqueeze(0)
        diff = torch.argmax(self.model1(x)) - torch.argmax(self.model2(x))
        return diff.item() == 0
    
    def tournament_select(self, population, model, target_size=None, tournament_size=4):
        if target_size is None:
            target_size = len(population)
        
        next_gen_pop = []
        fitness = np.array([self.fitness_function(x, model, ref=1) for x in population])
        best_idx = np.argmin(fitness)
        for i in range(target_size):
            if i == 0:
                next_gen_pop.append(population[best_idx])
            else:
                indices = [random.randint(0, len(population)-1) for _ in range(tournament_size)]
                sub_fitness = [fitness[idx] for idx in indices]
                best_sub_fitness = min(sub_fitness)
                for idx in indices:
                    if (fitness[idx] == best_sub_fitness):
                        next_gen_pop.append(population[idx])
                        break
        return next_gen_pop
        
    def genetic_algorithm(self, x):
        start = time.perf_counter()

        population = self.init_population(x)
        
        if self.iteration == -1:
            iteration = int(1e11)
        else:
            iteration = self.iteration
        for generation in range(iteration):
            if self.timeout > 0 and time.perf_counter() - start > self.timeout:
                if self.verbose:
                    print("Failed. Can not find any disagreements in limit time {:.3f} seconds. Return mutated.".format(self.timeout))
                return population[0]
            
            for sample in population:
                if not self.agree(sample):
                    if self.verbose:
                        print("Successed. Quit before gen {} with a disagreement in {:.3f} seconds.".format(generation, (time.perf_counter() - start)))
                    return sample
            
            pop1 = self.tournament_select(population, self.model1, len(population) // 2, 4)
            pop2 = self.tournament_select(population, self.model2, len(population) // 2, 4)
            two_best = [pop1[0], pop1[0]]
            other_population = pop1[1: ] + pop2[1: ]
            
            # 交叉
            cross_population = []
            for i in range(len(other_population) // 2):
                indices = [random.randint(0, len(other_population)-1) for _ in range(2)]
                x1, x2 = other_population[indices[0]], other_population[indices[1]]
                x1, x2 = self.crossover(x1, x2)
                cross_population.extend([x1, x2])
            # 变异
            population = [self.mutate(x) for x in cross_population]
            
            # 剩下的两个最好的补上
            population = two_best + population

        if self.verbose:
            print("Failed. Can not find any disagreements in limit {} iterations. Return mutated.".format(iteration))
        return population[0]
    
    def __call__(self, x, *args, **kwargs):
        """
        Args:
            x (_type_): 接受的输入是：(N, C, H, W)或(C, H ,W)

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        assert isinstance(x, torch.Tensor), "x need to be torch.Tensor."
        start = time.perf_counter()
        if x.ndim == 4:
            result = []
            for single_x in x:
                result.append(self.genetic_algorithm(single_x))
            result = torch.cat(result)
        elif x.ndim == 3:
            result = self.genetic_algorithm(x)
        else:
            raise ValueError()
        if self.verbose:
            print("Time cosumption: {:.3f} s".format(time.perf_counter() - start))
        return result