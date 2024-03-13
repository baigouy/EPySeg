from functools import wraps
import time

# use this class to wrap and time any function without modifying it

def timer(func):
    """Decorator to time functions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Inner wrapper function"""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(end-start)
        return result
    return wrapper

@timer
def func():
   print('bob')

@timer
def addition(a,b):
    return a+b


if __name__ == '__main__':
    func()

    res = addition(10,12)
