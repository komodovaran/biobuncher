from lib.utils import pairwise

array = ["a", "b", "c", "d", "e", "f", "g"]

array = list(range(len(array)))

if len(array) % 2 != 0:
    array.append(None)

for a, b in pairwise(array):
    print(a, b)