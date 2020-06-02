import math
import random
from collections import Counter
from matplotlib import pyplot as plt
import dateutil.parser
import csv

from probability import inverse_normal_cdf
from statistics import correlation
from linear_algebra import shape, get_column, make_matrix

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


#Processing Multy dimension data

def correlation_matrix(data):
    _, num_columns = shape(data)

    def matrix_entry(i, j):
        return correlation(get_column(data, i), get_column(data, j))
    
    return make_matrix(num_columns, num_columns, matrix_entry)


def run_multy_dimension_data_process():
    from typing import List

    # Just some random data to show off correlation scatterplots
    num_points = 100

    def random_row() -> List[float]:
       row = [0.0, 0, 0, 0]
       row[0] = random_normal()
       row[1] = -5 * row[0] + random_normal()
       row[2] = row[0] + row[1] + 5 * random_normal()
       row[3] = 6 if row[2] > -2 else 0
       return row

    random.seed(0)
    # each row has 4 points, but really we want the columns
    corr_rows = [random_row() for _ in range(num_points)]

    corr_data = [list(col) for col in zip(*corr_rows)]

    # corr_data is a list of four 100-d vectors
    num_vectors = len(corr_data)
    fig, ax = plt.subplots(num_vectors, num_vectors)

    for i in range(num_vectors):
        for j in range(num_vectors):

            # Scatter column_j on the x-axis vs column_i on the y-axis,
            if i != j: ax[i][j].scatter(corr_data[j], corr_data[i])

            # unless i == j, in which case show the series name.
            else: ax[i][j].annotate("series " + str(i), (0.5, 0.5),
                                    xycoords='axes fraction',
                                    ha="center", va="center")

            # Then hide axis labels except left and bottom charts
            if i < num_vectors - 1: ax[i][j].xaxis.set_visible(False)
            if j > 0: ax[i][j].yaxis.set_visible(False)

    # Fix the bottom right and top left axis labels, which are wrong because
    # their charts only have text in them
    ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    ax[0][0].set_ylim(ax[0][1].get_ylim())

    plt.show()

# Cleaning and formatting

def try_or_none(f):
    def f_or_none(x):
        try: return(f(x))
        except: return None
    return f_or_none

def parse_row(input_row, parsers):
    return [try_or_none(parser)(value) if parser is not None else value 
            for value, parser in zip(input_row, parsers)]

def parse_rows_with(reader, parsers):
    for row in reader:
        yield parse_row(row, parsers)
    

def run_parsing_exception_handling():
    data = []
    
    with open("data_science_hints/resources/comma_delimited_stock_prices.csv", 'r') as f:
        reader = csv.reader(f)
        for line in parse_rows_with(reader, [dateutil.parser.parse, None, float]):
            data.append(line)
        
        for row in data:
            if any(x is None for x in row):
                print(row)

def try_parse_field(field_name, value, parser_dict):
    parser = parser_dict.get(field_name)
    if parser is not None:
        return try_or_none(parser)(value)
    else:
        return value

def parse_dict(input_dict, parser_dict):
    return {field_name : try_parse_field(field_name, value, parser_dict)
            for field_name, value in input_dict.items()}

def run_dict_parser():
    input_dict = {"first": "0.98", "second": 6.8, "third": "not float"}
    parser_dict = {"first": float, "second": str, "third": float}
    print(parse_dict(input_dict, parser_dict))

if __name__ == "__main__":
    # run_single_demension_data_process()
    # run_two_demension_data_process()
    # run_multy_dimension_data_process()
    # run_parsing_exception_handling()
    run_dict_parser()
