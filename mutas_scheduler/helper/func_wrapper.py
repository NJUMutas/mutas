import time
import os
from pickle import dump, load


def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('%s spent %d s' % (func.__name__, end - start))
        return result

    return wrapper


def cache_wrapper(cache_filename):
    def decorator(func):
        def wrapper(*args, **kw):

            if not os.path.exists(cache_filename):
                obj = func(*args, **kw)
                with open(cache_filename, 'wb') as out:
                    dump(obj, out)
            else:
                with open(cache_filename, 'rb') as file:
                    obj = load(file)
            return obj

        return wrapper

    return decorator
