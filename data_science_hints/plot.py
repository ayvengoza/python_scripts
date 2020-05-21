from matplotlib import pyplot as plt
from collections import Counter


def linear_plot_ex():
    years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
    gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]
    plt.plot(years, gdp, color='green', marker='o', linestyle='solid')
    plt.title("Nominal GDP")
    plt.ylabel("Miliard $")
    plt.show()

def bar_plot_ex():
    movies = ['Holl', 'Gur', 'Cassa', 'Gandy', 'Westside']
    num_oscars = [5, 11, 3, 8, 10]
    xs = [i for i, _ in enumerate(movies)]
    plt.bar(xs, num_oscars)
    plt.ylabel("Num of oscars")
    plt.title("Films")
    plt.xticks([i for i, _ in enumerate(movies)], movies)
    plt.show()

def bar_frequency_plot_ex():
    grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]
    decline = lambda grade: grade // 10 * 10
    histogram = Counter(decline(grade) for grade in grades)
    plt.bar([x for x in histogram.keys()], histogram.values(), width=8)
    plt.axis([-5, 105, 0, 5])
    plt.xticks([10 * i for i in range(11)])
    plt.xlabel("Decile")
    plt.ylabel("Num of students")
    plt.title("grades frequency")
    plt.show()

def multy_linear_plot_ex():
    variance = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    bias_squered = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    total_error = [x + y for x, y in zip(variance, bias_squered)]
    xs = [i for i, _ in enumerate(variance)]
    plt.plot(xs, variance, 'g-', label='variance')
    plt.plot(xs, bias_squered, 'r-', label='bias^2')
    plt.plot(xs, total_error, 'b:', label='total error')
    plt.legend(loc=9)
    plt.xlabel('Dificult of model')
    plt.title("Compromice")
    plt.show()

def scatter_plot_ex():
    friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
    minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    plt.scatter(friends, minutes)
    for label, friend_count, minute_count in zip(labels, friends, minutes):
        plt.annotate(label, xy=(friend_count, minute_count), xytext=(5, -5), textcoords='offset points')

    plt.title("Minutes vs Friends")
    plt.xlabel("Num of friends")
    plt.ylabel("Minutes per day")
    plt.show()

def scatter_equal_plot_ex():
    test_1_grades = [99, 90, 85, 97, 80]
    test_2_grades = [100, 85, 60, 90 ,70]
    plt.scatter(test_1_grades, test_2_grades)
    plt.xlabel("Test 1")
    plt.ylabel("Test 2")
    plt.axis("equal") # normalize axis
    plt.grid()
    plt.show()

def __is_near__(nums, item, around):
    for n in nums:
        if(item < n + around and item > n -around):
            return True
    return False 


def bar_frequency_plot(num_arr, bar_count = 10, start = 0, end = 100):
    mod = abs((end - start)) // bar_count
    if mod == 0:
        mod = 1
    backet = lambda item: item // mod * mod + mod/2
    histogram = Counter(backet(item) for item in num_arr)
    xs = sorted([x for x in histogram.keys() if (x > (start) - mod)])
    ys = [histogram[x] for x in xs]
    plt.bar(xs, ys, width = mod * 0.8 )
    xtick_intervals = [int(t) for t in xs]
    xtick_intervals += ([int(t) - int(mod/2) for t in xs])
    xtick_intervals += ([int(t) + int(mod/2) for t in xs])
    xtick_intervals += [t for t in range(int(start), int(end) + 1, int(mod/2)) if not __is_near__(xs, t, mod)]
    plt.xticks(xtick_intervals)
    plt.xlabel("Subject")
    plt.ticklabel_format(useOffset=False)
    plt.axis([start, end, 0, max(histogram.values())])
    plt.ylabel("Num")
    plt.title("frequency")
    plt.show()


if __name__ == "__main__":
    scatter_equal_plot_ex()

