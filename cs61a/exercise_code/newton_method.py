import torch
def newton_method(f,df):
    """
    return a function,parameter is x
    the funtion return one iteration of x value
    """
    def calculate(x):
        return x - f(x)/df(x)
    return calculate


def improve(f, df, x = 1,tolerance=1e-10):
    while abs(f(x)) > tolerance:
        x = newton_method(f,df)(x)
    return x

def nth_root(n,a):
    f = lambda x:pow(x,n) - a
    df = lambda x: n*pow(x,n-1)

    return improve(f,df)

print("1:", torch.__version__)
