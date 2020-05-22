from functools import reduce
from math import sqrt

# Vectors

def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def vector_substruct(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors):
    result = vectors[0]
    for vector in vectors[1:]:
        result = vector_add(result, vector)
    return result

def vector_sum_reduce(vectors):
    return reduce(vector_add, vectors)

def scalar_multiply(c, v):
    """vector multiply on scalar"""
    return [c * v_i for v_i in v]

def vector_mean(vectors):
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def dot(v, w):
    """sum of vector multiply (scalar multiply)
    It`s projection of vector "v" on vector "w"
    length "v" in "w" direction """
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    return dot(v, v)

def magnitude(v):
    return sqrt(sum_of_squares(v))

def squered_distance(v, w):
    """(v_1 - w_1)^2 + ... + (v_n - w_n)^2"""
    return sum_of_squares(vector_substruct(v, w))

def distance(v, w):
    return sqrt(squered_distance(v, w))

def distance_2(v,w):
    return magnitude(vector_substruct(v, w))

# Matrix

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_rows

def get_row(A, i):
    return A[i]

def get_column(A, j):
    return [A_i[j] for A_i in A]

def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i, j)
            for j in range(num_cols)]
            for i in range(num_rows)]
            
def is_diagonal(i, j):
    return 1 if i == j else 0


if __name__ == "__main__":
    matrix = make_matrix(5, 5, is_diagonal)
    print(matrix)
