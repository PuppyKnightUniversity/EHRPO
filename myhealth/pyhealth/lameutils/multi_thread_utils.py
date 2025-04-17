import os
from time import *
import threading
import numpy as np


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)


def conver_line_contend(line):
    line = line.strip()
    if len(line) == 0:
        return None
    if line.startswith("INSERT INTO"):
        contend = "".join("".join(line.split("(")[1:]).split(")")[:-1]).split(", ")
        return [i[1:-1] for i in contend]
    return None


def async_kd_tokenizer(filename, worker_id, num_workers, filter_index, filter_vals, save_cols):
    with open(filename, 'r', encoding='utf-8') as f:
        size = os.fstat(f.fileno()).st_size  # 指针操作，所以无视文件大小
        # print(f'size {size}')
        chunk_size = size // num_workers
        offset = worker_id * chunk_size
        end = offset + chunk_size
        f.seek(offset)
        # print(f'offset {offset}')
        if offset > 0:
            safe_readline(f)  # drop first incomplete line
        lines = []
        line = f.readline()
        while line:
            contend_list = conver_line_contend(line)
            if contend_list is None:
                line = f.readline()
                continue
            cond_flag = True
            for iter, filter_item in enumerate(filter_index):
                if contend_list[filter_item] not in filter_vals[iter]:
                    cond_flag = False
                if not cond_flag:
                    break
            if cond_flag:
                lines.append(list(np.array(contend_list)[save_cols]))
            if f.tell() > end:
                break
            line = f.readline()
        return lines


# 多线程
class FileHandlerThread(threading.Thread):

    def __init__(self, func, args):
        super(FileHandlerThread, self).__init__()
        self.args = args
        self.func = func

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def encode_file_thread(path, workers=4, filter_index=None, filter_vals=None, save_cols=None):
    if filter_index is None:
        filter_index = []
    assert os.path.exists(path)
    results = []
    workers_thread = []
    for i in range(workers):
        w = FileHandlerThread(async_kd_tokenizer, args=(path, i, workers, filter_index, filter_vals, save_cols))
        workers_thread.append(w)
        w.start()
    for w in workers_thread:
        w.join()
    for w in workers_thread:
        result = w.get_result()
        results += result
    return results


def normal_read(file_path, line_num=10000):
    results = []
    line_c = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace(" ", '').replace("\n", '')
            line_c += 1
            if not line:
                continue
            if line_c >= line_num:
                break
            if "门诊" not in line and "医师" not in line and "门(急)诊" not in line:
                results.append(line)
    return results

####### test ########
# begin_time = time()
# results_th = encode_file_thread('H:/windows_materials/E/githubWorkSpace/medical_data/人民医院数据/lab_result.sql', workers=128, filter_index=[], filter_vals=[], save_cols=list(range(5)))
# print("result:", len(results_th))
# # print(results_th)
# end_time = time()
# print("time:", end_time-begin_time)
# begin_time = time()
# results_th = normal_read('../data_utils/code_charge_item.sql')
# print("result:", len(results_th))
# end_time = time()
# print("time:", end_time-begin_time)

# re = normal_read('H:/windows_materials/E/githubWorkSpace/medical_data/人民医院数据/billing_item.sql', line_num=50000000)
# for i in re:
#     print(i)
