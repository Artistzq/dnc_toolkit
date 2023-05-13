from functools import wraps
import warnings


def return_string(keys):
    def decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            res = func(*args, **kwargs)
            assert len(res) == len(keys)
            
            s = ""
            for key, r in zip(keys, res):
                s += "{}: {}\t".format(key, r)
            return s
        return wrapped_function
    return decorator


def deprecated(reason=None, new=None):
    def decorator(func):
        @wraps(func)
        def new_func(*args, **kwargs):
            message = "Call to deprecated function '{}'.".format(func.__name__)
            if reason:
                message += "\nThis method is deprecated now because of '{}'".format(reason)
            if new:
                message += "\nUse '{}' instead.".format(new)
            warnings.warn(message,
                          category=DeprecationWarning,
                          stacklevel=2)
            return func(*args, **kwargs)
        return new_func
    return decorator
