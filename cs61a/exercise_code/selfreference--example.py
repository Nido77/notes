cube = lambda x : x*x*x
def func(f):
    def input(x):
        print("the value of f(x):", f(x))
        return func(f)
    return input

cube_x = func(cube)
cube_x(3)
cube_x(4) 
cube_x(5)