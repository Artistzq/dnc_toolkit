from functools import wraps

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
