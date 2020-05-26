import random
import math
from matplotlib import pyplot as plt
from collections import Counter

def random_kid():
    return random.choice(["boy", "girl"])

def paradox_boy_girl():
    both_girls = 0
    older_girl = 0
    either_girl = 0

    random.seed(0)
    for _ in range(10000):
        younger = random_kid()
        older = random_kid()
        if older == "girl":
            older_girl += 1
        if older == "girl" and younger == "girl":
            both_girls += 1
        if older == "girl" or younger == "girl":
            either_girl += 1
    
    print("P(both | older):", both_girls / older_girl)
    print("P(both | either):", both_girls / either_girl)

def uniform_pdf(x):
    """Density of uniform probability"""
    return 1 if x >= 0 and x < 1 else 0

def uniforn_cdf(x):
    """Cumulative distribution function
    or Integral distribution probability"""
    if x < 0:   return 0
    elif x < 1: return x
    else:       return 1


def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

def plot_normal_pdf():
    xs = [x / 10.0 for x in range(-50, 50)]
    ys1 = [normal_pdf(x, sigma=1) for x in xs]
    ys2 = [normal_pdf(x, sigma=2) for x in xs]
    ys3 = [normal_pdf(x, sigma=0.5) for x in xs]
    ys4 = [normal_pdf(x, mu=-1) for x in xs]
    plt.plot(xs, ys1, '-', label='mu=0, sigma=1')
    plt.plot(xs, ys2, '--', label='mu=0, sigma=2')
    plt.plot(xs, ys3, ':', label='mu=0, sigma=0.5')
    plt.plot(xs, ys4, '-.', label='mu=-1, sigma=1')
    plt.legend()
    plt.title("Several PDF normal density distribution")
    plt.grid()
    plt.show()

def normal_cdf(x, mu=0, sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def plot_normal_cdf():
    xs = [x / 10.0 for x in range(-50, 50)]
    ys1 = [normal_cdf(x, sigma=1) for x in xs]
    ys2 = [normal_cdf(x, sigma=2) for x in xs]
    ys3 = [normal_cdf(x, sigma=0.5) for x in xs]
    ys4 = [normal_cdf(x, mu=-1) for x in xs]
    plt.plot(xs, ys1, '-', label='mu=0, sigma=1')
    plt.plot(xs, ys2, '--', label='mu=0, sigma=2')
    plt.plot(xs, ys3, ':', label='mu=0, sigma=0.5')
    plt.plot(xs, ys4, '-.', label='mu=-1, sigma=1')
    plt.legend(loc=4)
    plt.title("Several CDF normal distribution")
    plt.grid()
    plt.show()

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    if mu != 0 or sigma != 1 :
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z, low_p = -10.0, 0
    hi_z, hi_p = 10.0, 1

    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            hi_z, hi_p = mid_z, mid_p
        else:
            break

    return mid_z

def plot_inverse_normal_cdf():
    ps = [p / 100.0 for p in range(0, 101)]
    ys1 = [inverse_normal_cdf(p, sigma=1) for p in ps]
    ys2 = [inverse_normal_cdf(p, sigma=2) for p in ps]
    ys3 = [inverse_normal_cdf(p, sigma=0.5) for p in ps]
    ys4 = [inverse_normal_cdf(p, mu=-1) for p in ps]
    plt.plot(ps, ys1, '-', label='mu=0, sigma=1')
    plt.plot(ps, ys2, '--', label='mu=0, sigma=2')
    plt.plot(ps, ys3, ':', label='mu=0, sigma=0.5')
    plt.plot(ps, ys4, '-.', label='mu=-1, sigma=1')
    plt.legend(loc=4)
    plt.title("Several CDF normal distribution (inverse way)")
    plt.grid()
    plt.show()

def bernouli_trial(p):
    return 1 if random.random() < p else 0

def binomial(n, p):
    return sum(bernouli_trial(p) for _ in range(n))

def plot_hist_binomial_distriburion(p, n, num_points):
    data = [binomial(n, p) for _ in range(num_points)]

    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')

    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))

    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]
    plt.plot(xs, ys)
    plt.title("Binomial distribution and normal approximation of them")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    paradox_boy_girl()
    plot_normal_pdf()
    plot_normal_cdf()
    plot_inverse_normal_cdf()
    plot_hist_binomial_distriburion(0.75, 100, 10000)