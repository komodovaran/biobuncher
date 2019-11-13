a = [1, 2, 3, 4, 5, 6, 7, 9]
arga = "a" * len(a)
argb = "b" * len(a)

for i in zip(a, arga, argb):
    print(i)