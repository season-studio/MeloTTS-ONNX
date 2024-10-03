import tqdm
import time
def do_repeat(count: int, progress_bar=None, progress_title=None):
    def fn(func):
        ret_list = []
        for i in (tqdm.tqdm(range(count), desc=progress_title) if progress_bar is None else progress_bar(range(count), title=progress_title)):
            ret = func(step=i, count=count)
            ret_list.append(ret)
        return ret_list
    return fn

def measure_time(fn):
    def wrapper(*args, **kwargs):
        start_t = time.perf_counter()
        ret = fn(*args, **kwargs)
        end_t = time.perf_counter()
        return ret, (end_t - start_t)
    return wrapper

def get_avg_value(list, start_index=0):
    v = [x for _,x in list[start_index:]]
    return 0 if len(v) == 0 else (sum(v) / len(v))