import math
import random
from collections import Counter
from matplotlib import pyplot as plt

from probability import inverse_normal_cdf
from statistics import correlation

#Processing Single dimension data

def bucketize(point, bucket_size):
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points, bucket_size):
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points, bucket_size, title=""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.grid()
    plt.show()

def run_single_demension_data_process():
    random.seed(0)

    uniform = [200 * random.random() - 100 for _ in range(10000)]
    normal = [57 * inverse_normal_cdf(random.random())
                for _ in range(10000)]
    
    plot_histogram(uniform, 10, "Uniform values")
    plot_histogram(normal, 10, "Normal values")


#Processing Two dimension data

def random_normal():
    return inverse_normal_cdf(random.random())

def run_two_demension_data_process():
    xs = [random_normal() for _ in range(1000)]
    ys1 = [ x + random_normal() / 2 for x in xs]
    ys2 = [-x + random_normal() / 2 for x in xs]

    plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
    plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
    plt.ylabel('xs')
    plt.legend(loc=9)
    plt.title("Diferent distribution")
    plt.grid()
    plt.show()

    print(correlation(xs, ys1))
    print(correlation(xs, ys2))


#Processing Many dimension data



if __name__ == "__main__":
    # run_single_demension_data_process()
    run_two_demension_data_process()
