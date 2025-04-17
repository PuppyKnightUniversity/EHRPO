import os
import numpy as np
# import cPickle as CPickle
import pickle

def read_file_2dict(file_name, index_num):
    result_dict = {}
    with open(file_name, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line_sp = line.strip().split("\t")
            if index_num < len(line_sp):
                result_dict[line_sp[index_num]] = [x for it_x, x in enumerate(line_sp) if it_x != index_num]
            else:
                return None
    return result_dict

def read_file_2list(file_name, sep="\t"):
    result_list = []
    line_num = 0
    with open(file_name, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            if line_num > 0:
                line_sp = line.strip().split(sep)
                result_list.append(line_sp)
            line_num += 1
    return result_list

def write_list2csv(list_data, csv_path, file_name, sep="\t"):
    with open(os.path.join(csv_path, file_name), 'w') as f:
        for item in list_data:
            f.write(sep.join(item))
            f.write('\n')


def write2pickle(file_path, object_list, type):
    with open(file_path, type) as f:
        for obj in object_list:
            pickle.dump(obj, f)
        f.close()


def read_pickle(file_path, type, num):
    obj_list = []
    with open(file_path, type) as f:
        for i in range(num):
            obj_list.insert(0, pickle.load(f))
        f.close()
    return obj_list


def save_set_dict_2file(list_x, file_path, file_name, index=True):
    with open(file_path + file_name, 'w')as f:
        for i, item in enumerate(list_x):
            if index:
                f.write(str(i) + '\t')
            f.write(str(item) + '\n')
    print("write file done.")

