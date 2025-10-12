def factorial_iter(num):
    from math import prod
    return 1 if num < 2 else prod(range(2, num+1))
