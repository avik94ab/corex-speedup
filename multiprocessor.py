from multiprocessing import Pool
from itertools import repeat

def sum_four(a, b, c, d):
    return a + b + c + d

a, b, c = 1, 2, 3

all_d_values = [1, 2, 3, 4]

print(list(repeat(a,4)))



