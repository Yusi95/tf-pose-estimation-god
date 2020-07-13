import math
import datetime
import multiprocessing as mp


def train_on_parameter(name, param):
    result = 0
    for num in param:
        result += math.sqrt(num * math.tanh(num) / math.log2(num) / math.log10(num))

    return {name: result}


if __name__ == '__main__':
    pool = mp.Pool(8)

    param_dict = {
        'task1': list(range(10, 30000000)),
        'task2': list(range(30000000, 60000000)),
        'task3': list(range(60000000, 90000000)),
        'task4': list(range(90000000, 120000000)),
        'task5': list(range(120000000, 150000000))
    }

    results = [pool.apply_async(train_on_parameter, args=(name, param)) for name, param in param_dict.items()]

    final_res = [p.get() for p in results]

    print('hi')
