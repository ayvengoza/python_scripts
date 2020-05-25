from collections import Counter
from math import sqrt

from linear_algebra import dot


def mean(x):
    return sum(x) / len(x)

def median(v):
    """return mid value"""
    n = len(v)
    sorted_v = sorted(v)
    midpoint = n // 2
    if n % 2 == 1:
        return sorted_v[midpoint]
    else:
        lo = midpoint - 1
        hi = midpoint
        return (sorted_v[hi] - sorted_v[lo]) / 2

def quantile(x, p):
    """return value from x list witch corespond to p%"""
    p_index = int(p * len(x))
    return sorted(x)[p_index]

# delta in mid with 2/4 of size (1/4 , 2/4, 1/4)
def interquartile_range(x):
    return quantile(x, 0.75) - quantile(x, 0.25)

def mode(x):
    """return list, modes can be more than one"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]

def data_range(x):
    return max(x) - min(x)

def de_mean(x):
    """return deviation vector (list), substract mean from each"""
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

 
#(n-1) selective variance n>30,
#if use  n - it is general variance.
#In huge data list - x_bar is not accurate and use (n-1)
def variance(x):
    """dispersion or variance, avarage from sum of square deviation"""
    n = len(x)
    deviation = de_mean(x)
    sum_of_squares = sum([d_i ** 2 for d_i in deviation])
    return sum_of_squares / (n-1)

def standart_deviation(x):
    """square root from variance"""
    return sqrt(variance(x))

def covariance(x, y):
    """return constant witch defined dependency between two array of data.
    Value from - infinity to + infinity. If 0 - linear dependency is absent"""
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)

def correlation(x, y):
    """return constant witch defined dependency between two array of data.
    Value from -1 to +1. If 0 - linear dependency is absent"""
    stdev_x = standart_deviation(x)
    stdev_y = standart_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y

if __name__ == "__main__":
    num_friends = [100,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    daily_minutes = [1,68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]
    x = [-2, -1, 0, 1, 2]
    y = [2, 1, 0, 1, 2]
    y2 = [99.98, 99.99, 100, 100.01, 100.02]
    print(correlation(x, y2))
