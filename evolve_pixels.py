#!/usr/bin/env python3

from datetime import datetime
from tqdm import tqdm
import click
from time import sleep
import numpy as np
from numpy import random
import cv2
import matplotlib.pyplot as plt

target_image = np.array([], dtype=np.uint8)


class Gene:
    def __init__(self):
        self.img = np.zeros(target_image.size, dtype=np.uint8)
        self.score = 0
        self.fitness = 0

    def crossover(self, other):
        child1 = Gene()
        child2 = Gene()
        point = random.randint(0, len(target_image))
        child1.img = np.concatenate((self.img[:point], other.img[point:]), axis=0)
        child2.img = np.concatenate((other.img[:point], self.img[point:]), axis=0)

        return child1, child2

    def mutate(self, mutation_rates):
        rand_values = random.random(size=target_image.shape)  # speed drawback..
        mut_indices = rand_values < mutation_rates
        self.img[mut_indices] += random.randint(-8, 9, size=np.count_nonzero(mut_indices))
        self.img = np.clip(self.img, 0, 255)
        # for i in range(len(target_image)):
        #    if random.random() < mutation_rate:
        #        self.img[i] = random.randint(0, 1)

    @classmethod
    def initialize(cls):
        new_gene = cls()
        new_gene.img = random.randint(0, 256, len(target_image))

        return new_gene

    def calc_score(self):
        diff = self.img - target_image
        return len(target_image) * 255 - np.sum(np.abs(diff))


@click.command()
@click.option('-ps', '--pop-size', default=512, help="the size of the population of generated images")
@click.option('-gc', '--gen-count', default=-1, help="the amount of generations (iter) the algorithm shall run for")
@click.option('-mut', '--mut-rate', default=0.01, help="the rate of the gene mutation")
@click.option('-l', '--image-len', default=100, help="the length of the searched randomly generated image, "
                                                     "if 'image' is set this will be ignored!")
@click.option('-img', '--image', help="the path to the image that the genetic algorithm tries to replicate, "
                                      "if set 'img-len' will be ignored!")
@click.option('-s', '--silence', is_flag=True, help="silences the output fully")
@click.option('-v', '--verbose', is_flag=True, help="shows verbose information for the algorithmic process")
@click.option('-plt', '--plotting', is_flag=True, help="plots the current score of the top gene")
def main(pop_size: int, gen_count: int, mut_rate: float, image_len: int, image: str, silence: bool, verbose: bool,
         plotting: bool):
    """Simple program that runs a genetic algorithm over a population of randomly generated images
       with the task to optimise these images so that it matches the provided target image"""

    setup(image, image_len)
    x = []
    y = []
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    li, = ax.plot(x, y)

    if plotting:
        # draw and show it
        fig.canvas.draw()
        plt.show(block=False)

    if not silence:
        print("image length: {}".format(str(len(target_image))))
        print("population size: {}".format(str(pop_size)))
        print("mutation rate: {}".format(str(mut_rate)))
        print()

    start = datetime.now()
    pop = []

    for i in range(pop_size):
        pop.append(Gene.initialize())

    mutation_rates = np.array([mut_rate] * len(target_image))

    top_gene = Gene()
    gen = 0
    with tqdm(disable=silence or verbose) as t:
        while top_gene.score != len(target_image):  # each generation
            if gen_count != -1 and gen == gen_count:
                break

            gen += 1

            if not silence:
                t.set_postfix(str="score:{}/{}".format(str(top_gene.score), str(len(target_image) * 255)))
                t.update()

            score_sum = 0
            for gene in pop:  # each gene
                gene.score = gene.calc_score()
                score_sum += gene.score

            top_gene = pop[0]
            for gene in pop:  # calc fitness
                gene.fitness = gene.score / score_sum
                if gene.fitness > top_gene.fitness:
                    top_gene = gene

            current = datetime.now()

            if not silence:
                img = np.array(top_gene.img.reshape(img_shape), dtype=np.uint8)
                target_img = np.array(target_image.reshape(img_shape), dtype=np.uint8)
                vis = np.concatenate((img, target_img), axis=1)
                vis = cv2.resize(vis, (vis.shape[1] * 4, vis.shape[0] * 4), interpolation=cv2.INTER_AREA)
                cv2.imshow("gen", vis)
                cv2.waitKey(1)
            if verbose:
                print("{:10.2f}ms - {:5} gens - score {}/{}".format((current - start).total_seconds() * 1000,
                                                                    gen, top_gene.score, len(target_image) * 255))
            if plotting:
                x.append(gen)
                y.append(top_gene.score)
                li.set_xdata(x)
                li.set_ydata(y)
                ax.relim()
                ax.autoscale_view(True, True, True)
                fig.canvas.draw()

            new_pop = []
            for k in range(int(len(pop) / 2)):  # each gene
                child_indices = random.randint(0, len(pop), 4)
                children = [pop[index] for index in child_indices]
                child1 = max(children[2:], key=lambda c: c.fitness)
                child2 = max(children[:2], key=lambda c: c.fitness)
                # for i in range(2):  # 2 times
                #     rand = random.random()
                #     sum_up = 0
                #     for j in range(pop_size):
                #         sum_up += pop[j].fitness
                #         if sum_up > rand:
                #             children.append(pop[j])
                #             break
                child1, child2 = child1.crossover(child2)
                child1.mutate(mutation_rates)
                child2.mutate(mutation_rates)
                new_pop.append(child1)
                new_pop.append(child2)

            pop[random.randint(0, len(pop))] = top_gene
            pop = new_pop

    t.close()
    end = datetime.now()

    sleep(1)
    print("{:10.2f}ms - {:5} gens - score {}/{}".format((end - start).total_seconds() * 1000, gen,
                                                        top_gene.score,
                                                        len(target_image)))


def setup(image, image_len):
    global target_image
    global img_shape

    # change phrase
    if image is not None:
        target_image = cv2.imread(image, flags=cv2.IMREAD_GRAYSCALE)
        img_shape = target_image.shape
        target_image = target_image.flatten()
    else:  # or generate it
        target_image = np.array(list([random.randint(0, 256) for _ in range(image_len)]))
        l = int(np.sqrt(len(target_image)))
        img_shape = (l, l)


if __name__ == "__main__":
    main()
