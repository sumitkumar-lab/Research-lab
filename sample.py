def square(x):
    return x**2

f = square

def function(fun, x, y):
    return fun(x) + fun(y)

print(function(f, 4, 5))

list = [1, 2, 3, 4]

result = map(f, (1, 2, 3))
print(result)
