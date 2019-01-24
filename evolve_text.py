#!/usr/bin/env python3

from datetime import datetime
import random
import string
from tqdm import tqdm
import click
from time import sleep

text_sequence = []
alphabet = string.ascii_letters + " " + string.digits + "!,.?#:-_<>;'*+~"


def rand_char():
    return random.choice(alphabet)


class Gene:
    def __init__(self):
        self.text = []
        self.score = 0
        self.fitness = 0

    def crossover(self, other):
        child1 = Gene()
        child2 = Gene()
        point = random.randint(0, len(text_sequence))
        child1.text = self.text[:point] + other.text[point:]
        child2.text = other.text[:point] + self.text[point:]

        return child1, child2

    def mutate(self, mutation_rate):
        for i in range(len(text_sequence)):
            if random.random() < mutation_rate:
                self.text[i] = rand_char()

    @classmethod
    def initialize(cls):
        new_gene = cls()
        for i in range(len(text_sequence)):
            new_gene.text.append(rand_char())

        return new_gene

    def calc_score(self):
        same = 0
        for i in range(len(text_sequence)):
            if self.text[i] == text_sequence[i]:
                same += 1

        return same


@click.command()
@click.option('-ps', '--pop-size', default=512, help="the size of the population of generated text sequences")
@click.option('-gc', '--gen-count', default=-1, help="the amount of generations (iter) the algorithm shall run for")
@click.option('-mut', '--mut-rate', default=0.01, help="the rate of the gene mutation")
@click.option('-l', '--phrase-len', default=100, help="the length of the searched randomly generated text sequence, "
                                                        "if 'phrase' is set this will be ignored!")
@click.option('-p', '--phrase', help="the phrase that the genetic algorithm is searching for, if set 'phrase-len' will "
                                     "be ignored!")
@click.option('-sa', '--small-alphabet', is_flag=True, help="defines the type of the used alphabet")
@click.option('-s', '--silence', is_flag=True, help="silences the output fully")
@click.option('-v', '--verbose', is_flag=True, help="shows verbose information for the algorithmic process")
def main(pop_size: int, gen_count: int, mut_rate: float, phrase_len: int, phrase: str, small_alphabet: bool,
         silence: bool, verbose: bool):
    """Simple program that runs a genetic algorithm over a population of randomly generated text sequences
       with the task to optimise the text so that it matches the searched text"""

    setup(phrase, phrase_len, small_alphabet)

    if not silence:
        print("searching for word: {}".format("".join(text_sequence)))
        print("phrase length: {}".format(str(len(text_sequence))))
        print("population size: {}".format(str(pop_size)))
        print("mutation rate: {}".format(str(mut_rate)))
        print("using alphabet: {}".format(alphabet))
        print("alphabet size: {}".format(str(len(alphabet))))
        print()

    start = datetime.now()
    pop = []

    for i in range(pop_size):
        pop.append(Gene.initialize())

    top_gene = Gene()
    gen = 0
    with tqdm(disable=silence or verbose) as t:
        while top_gene.score != len(text_sequence):  # each generation
            if gen_count != -1 and gen == gen_count:
                break

            gen += 1

            if not silence:
                t.set_postfix(str="score:{}/{}".format(str(top_gene.score), str(len(text_sequence))))
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
            if verbose:
                print("{:10.2f}ms - {:5} gens - score {}/{} - text: {} ".format((current - start).total_seconds() * 1000,
                                                                                gen, top_gene.score, len(text_sequence),
                                                                                "".join(top_gene.text)))

            new_pop = []
            for k in range(int(len(pop) / 2)):  # each gene
                child1 = max(random.choices(population=pop, k=2), key=lambda x: x.fitness)
                child2 = max(random.choices(population=pop, k=2), key=lambda x: x.fitness)
                # for i in range(2):  # 2 times
                #     rand = random.random()
                #     sum_up = 0
                #     for j in range(pop_size):
                #         sum_up += pop[j].fitness
                #         if sum_up > rand:
                #             children.append(pop[j])
                #             break
                child1, child2 = child1.crossover(child2)
                child1.mutate(mut_rate)
                child2.mutate(mut_rate)
                new_pop.append(child1)
                new_pop.append(child2)

            pop[random.randint(0, len(pop) - 1)] = top_gene
            pop = new_pop
        t.close()
        end = datetime.now()

    sleep(1)
    print("{:10.2f}ms - {:5} gens - score {}/{} - text: {} ".format((end - start).total_seconds() * 1000, gen,
                                                                    top_gene.score,
                                                                    len(text_sequence), "".join(top_gene.text)))


def setup(phrase, phrase_len, small_alphabet):
    global text_sequence
    global alphabet

    # change alphabet
    if small_alphabet:
        alphabet = string.ascii_lowercase + " "

    # change phrase
    if phrase is not None:
        text_sequence = phrase
    else:  # or generate it
        text_sequence = list([rand_char() for _ in range(phrase_len)])

    # check if alphabet contains all characters of the phrase
    for char in text_sequence:
        assert alphabet.__contains__(char), "character {} is not available in this alphabet:\n{}".format(char, alphabet)


if __name__ == "__main__":
    main()
