import os
from time import *
import numpy as np
from multiprocessing import Pool
from utils.nlp_utils import NlpUtility
import difflib

def test1(**args):
    for item in args:
        print(item, args[item])

def test2(**args):
    test1(w="fsaf", r="wefwe", **args)


def __safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)


def __conver_line_contend(line):
    line = line.strip()
    if len(line) == 0:
        return None
    if line.startswith("INSERT INTO"):
        contend = "".join("".join(line.split("(")[1:]).split(")")[:-1]).split(", ")
        return [i[1:-1] for i in contend]
    return None


def __norm_line_split(line, sp="\t"):
    # line = line.strip()
    if len(line) == 0:
        return None
    contend = [i.strip() for i in line.split(sp)]
    return contend


def __change_line(line, change_type, change_col, change_val=None):
    if change_type == "INSERT":
        if change_col > len(line) or change_col < -1 * len(line):
            print("INSERT ERROR!, Insert column is out of the length of line!")
        else:
            line.insert(change_col, change_val)
    elif change_type == "DELETE":
        if change_col >= len(line) or change_col < -1 * len(line):
            print("DELETE ERROR!, Delete column is out of the length of line!")
        else:
            del line[change_col]
    elif change_type == "DEL_EMP":
        new_line = []
        for i in range(len(line)):
            if len(line[i]) != 0:
                new_line.append(line[i])
        line = new_line
    elif change_type == "CHANGE":
        if isinstance(change_col, int):
            change_col = [change_col]
            change_val = [change_val]
        for change_col_i in range(len(change_col)):
            if change_col[change_col_i] >= len(line) or change_col[change_col_i] < -1 * len(line):
                print("CHANGE ERROR!, Change column is out of the length of line!")
            else:
                line[change_col[change_col_i]] = change_val[change_col_i]
    else:  # change_type == "COND_CHANGE"
        if isinstance(change_col, int):
            change_col = [change_col]
            change_val = [change_val]
        for change_col_i in range(len(change_col)):
            if change_col[change_col_i] >= len(line) or change_col[change_col_i] < -1 * len(line):
                print("CHANGE ERROR!, Change column is out of the length of line!")
            else:
                change_cond = change_val[change_col_i]
                change_key = line[change_col[change_col_i]]
                if change_key in change_cond:
                    line[change_col[change_col_i]] = change_cond[change_key]
    return line


def preprocess_labtest_sub_6_1_get_func(line_contend):
    # 6.1 for num_qual_f_diff_ch file
    return [line_contend[-1], line_contend[0], "无法判定"]


def preprocess_labtest_sub_6_2_get_func(line_contend, nqsc_norm_dict):
    # 6.2 for num_qual_same_ch file
    nqsc_res = [line_contend[-1], line_contend[0]]
    # print(item[2], type(item[2]))
    neg_res = {'阴性', '未见', 'RBC阴性', '血型复检结果相符', '阴性，RO52+', '阴性,RO52+', 'O  型', 'B  型', 'A  型'}
    pos_res = {'阳性', '弱阳性', '阳性1:100', '阳性+++', '1:10阳性', '阳性+', '阳性++', '阳性（+）', '极弱阳性', '镜下+',
               '1:40阳性', '1:160阳性', ' 阳性1:1'}
    level_res = {'<1:40': 40, '<1:80': 80, '<1:160': 160, '<1:10': 10}
    val_tem = {'1:40斑点': 40, '1:20斑点型': 20, '1:320斑点': 320, '1:640斑点': 640, '1:80均质型': 80, '1:40均质型': 40, '1:80斑点': 80,
               '1:160均质': 160,
               '1:160斑点': 160, '1:80均质型,斑点型': 80, '1:320着丝点': 320, '1:40均质型,斑点型': 40, '1:640斑点/胞浆': 640,
               '1:160均质型': 160,
               '1:20均质型': 20, '小于0.02': 20, '1:320核点': 320, '1:160核膜/斑点': 160, '1:80斑点型': 80, '1:80胞浆颗粒型': 80,
               '1:320胞浆/斑点': 320,
               '1:80核仁型,斑点型': 80, '1:40均质': 40, '1:320核仁': 320, '1:640斑点型': 640, '1:20核仁型,均质型': 20, '1:80斑点型,核仁型': 80,
               '1:80着丝点/均质': 80, '1:320均质': 320, '1:80均质型,核仁型': 80, '1:40均质/核仁': 40, '1:320斑点/均质': 320, '1:80均质': 80}

    if line_contend[1] in neg_res:
        nqsc_res.append('阴性')
    else:
        if line_contend[1] in pos_res:
            nqsc_res.append('阳性')
        else:
            if line_contend[1] == line_contend[5] and line_contend[5] == '野生型':
                nqsc_res.append('阴性')
            else:
                if line_contend[5] == '未见' and line_contend[5] != line_contend[1]:
                    nqsc_res.append('阳性')
                else:
                    if line_contend[1] in val_tem:
                        if line_contend[0] in nqsc_norm_dict:
                            item5 = nqsc_norm_dict[line_contend[0]]
                        else:
                            item5 = line_contend[5]
                        if item5 in level_res:
                            if val_tem[line_contend[1]] >= level_res[item5]:
                                nqsc_res.append('阴性')
                            else:
                                nqsc_res.append('阳性')
                        else:
                            nqsc_res.append('无法判定')
                    else:
                        nqsc_res.append('无法判定')
    return nqsc_res


def preprocess_labtest_sub_6_3_get_func(line_contend, nqse_norm_dict):
    # 6.3 for num_qual_same_en file
    nqse_res = [line_contend[-1], line_contend[0]]
    level_map = {'<1E+03': "--1000", '<5E+02': "--500", '<5.00E+2': "--500", '0': "0--0", '<1:10': "10--"}
    neg_res = {'-', '-----', '—', 'B', 'A', 'O', 'AB'}
    pos_res = {'4+', '+', '±', '++', '+++', '3+', '1+', '++++', '+-', '+/-', '+++++'}
    num_res = {'0.361', '3.99E +04', '2.28E+ 3', '5.29E+ 2', '7.54E+ 3', '5.16E+ 2', '1.57E+ 3', '4.03E+ 2',
               '2.50E+ 4', '5.00E+ 3', '1.50E+ 3', '5.07E+ 3', '5.92E+ 2', '8.95E+ 2', '8.73E+ 2', '2.39E+ 4',
               '6.38E+ 2', '1.57E+ 4', '9.23E+ 2', '2.04E+ 5', '8.32E+ 2', '4.75E+ 2', '9.96E+ 2', '1.34E+ 3',
               '3.14E+ 2', '2.86E+ 4', '1.60E+ 3', '7.80E+ 2', '1.52', '0.471', '1.12', '1.06E+ 6', '4.91E+ 2',
               '3.08E+ 3', '5.86E+ 3', '3.26E+ 3', '1.68E+ 6', '6.06E+ 4', '0.10E+ 06', '5.72E+ 2', '1.24E +5',
               '1.93E+ 3', '0.20E+ 06', '8.97E+ 2', '1.32E+ 5', '6.12E+ 2', '8.37E+ 2', '1.57E+ 7', '3.81E+ 7',
               '6.85E+ 5', '1.24E+ 4', '8.09E+ 2', '1.38E+ 4', '2.10E+ 3', '5.54E +02', '4.37E+ 3', '2.99E+ 06',
               '3.28E+ 4', '1.71E+ 3', '3.66E+ 7', '1.47E+ 3', '0.12E+ 06', '9.05E+ 4', '4.90E+ 3', '1.87E+ 4',
               '7.82E+ 2', '5.48E+ 3', '3.87E +02', '1.20E+ 3', '2.82', '1.012', '52.05', '1.22', '1.82E+ 3',
               '8.79E +02', '2.43E +03', '2.17E+ 3', '5.90E+ 3', '7.24E+ 2', '0.08E+ 06', '1.56E+ 8', '2.77E+ 4',
               '1.53E+ 4', '1.88E+ 4', '8.96E+ 3', '1.97E+ 4', '2.39E+ 3', '2.43E+ 3', '0.21E+ 06', '9.40E +03',
               '2.58E+ 3', '1.71E+ 4', '2.38E+ 3', '1.52E+ 4', '1.80E+ 6', '2.33E+ 3', '4.09E+ 3', '2.59E+ 3',
               '1.66E+ 3', '2.56E+ 3', '8.31E+ 4', '3.08E+ 5', '4.95E+ 3', '2.87E+ 3', '1.65E+ 3', '2.51E+ 3'}
    range_res = {'0-1', '0-3', '1-6', '5-10', '10-15'}
    level_res = {'1:32': 32, '1:64': 64, '1:16': 16, '1:40': 40, '1:20': 20, '1:8': 8, '1:128': 128, '1:4': 4,
                 '1:256': 256, '1:80': 80, '1:2': 2, '1:640': 640, '1:512': 512, '<1:20': 20}
    half_range = {'<5E+02', '<1E+03', '< 0.350', '<20.0', '<=1.005', '<<8.0', '<<3.0', '>=1.030', '<5.00E+ 2',
                  '< 15.0', '<<10.0', '<0.1', '<2', '< 28.0', '>127.21', '<25.0', '<<0.50', '<0.001',
                  '> 250.00', '<0.600', '> 1300.0', '> 1000.00', '<5.0', '<0.300', '<1.85', '<0.200', '>20.00',
                  '>18.576', '>200000', '<1.00', '> 500.0', '>2000', '<15.0', '>8.00', '>97.000', '>=55', '<0.003',
                  '>308.0', '<15', '>5000', '<5.00', '>20000', '>9000', '>140', '>>30.0', '>4.8', '<<5.0', '<30.0',
                  '<0.0667', '>50', '>1000', '>1210', '>25', '>9', '>500', '<0.0507', '>210', '>>180', '<11.1',
                  '<0.0580', '>150', '>200.00', '<1', '<0.00', '<<0.5', '< 0.30', '> 30.0', '<0.010', '<1.0', '<5.83',
                  '<0.0412', '>3814.0', '> 150.000', '>180'}
    item = line_contend
    if item[1] in neg_res:
        nqse_res.append('阴性')
    else:
        if item[1] in pos_res:
            nqse_res.append('阳性')
        else:
            if item[1] in num_res:
                item_num = float(item[1].replace(" ", ""))
                if item[0] in nqse_norm_dict:
                    item_range = nqse_norm_dict[item[0]].split("--")
                    flag = "阴性"
                    if len(item_range[0]) > 0:
                        if item_num < float(item_range[0]):
                            flag = "阳性"
                        else:
                            if len(item_range[1]) > 0:
                                if item_num > float(item_range[1]):
                                    flag = "阳性"
                    else:
                        if len(item_range[1]) > 0:
                            if item_num > float(item_range[1]):
                                flag = "阳性"
                    nqse_res.append(flag)
                else:
                    nqse_res.append("无法判定")
            else:
                if item[1] in range_res:
                    item_tsp = item[1].split("-")
                    low_val = float(item_tsp[0])
                    high_val = float(item_tsp[1])
                    if item[0] in nqse_norm_dict:
                        item_range = nqse_norm_dict[item[0]].split("--")
                        flag = "阴性"
                        if len(item_range[0]) > 0:
                            if low_val < float(item_range[0]):
                                flag = "阳性"
                            else:
                                if len(item_range[1]) > 0:
                                    if high_val > float(item_range[1]):
                                        flag = "阳性"
                        else:
                            if len(item_range[1]) > 0:
                                if high_val > float(item_range[1]):
                                    flag = "阳性"
                        nqse_res.append(flag)
                    else:
                        nqse_res.append("无法判定")
                else:
                    if item[1] in level_res:
                        item_num = level_res[item[1]]
                        if item[0] == "12":
                            item_range = nqse_norm_dict[item[0]].split("--")
                            flag = "阴性"
                            if len(item_range[0]) > 0:
                                if item_num < float(item_range[0]):
                                    flag = "阳性"
                                else:
                                    if len(item_range[1]) > 0:
                                        if item_num > float(item_range[1]):
                                            flag = "阳性"
                            else:
                                if len(item_range[1]) > 0:
                                    if item_num > float(item_range[1]):
                                        flag = "阳性"
                            nqse_res.append(flag)
                        else:
                            nqse_res.append("无法判定")
                    else:
                        if item[1] in half_range:
                            item_num = float(
                                item[1].replace(" ", "").replace("<", "").replace(">", "").replace("=", ""))
                            if item[0] in nqse_norm_dict:
                                item_range = nqse_norm_dict[item[0]].split("--")
                                flag = "阴性"
                                if len(item_range[0]) > 0:
                                    if item_num < float(item_range[0]):
                                        flag = "阳性"
                                    else:
                                        if len(item_range[1]) > 0:
                                            if item_num > float(item_range[1]):
                                                flag = "阳性"
                                else:
                                    if len(item_range[1]) > 0:
                                        if item_num > float(item_range[1]):
                                            flag = "阳性"
                                nqse_res.append(flag)
                            else:
                                nqse_res.append("无法判定")
                        else:
                            nqse_res.append("无法判定")
    return nqse_res


def preprocess_labtest_sub_6_4_get_func(line_contend, nf_norm_dict):
    # 6.4 for num_float file
    nf_res = [line_contend[-1], line_contend[0]]
    item = line_contend
    if ":" not in item[1] and "-." not in item[1] and not NlpUtility.has_chinese_char(item[1]):
        if "-" in item[1]:
            item1_sp = item[1].split("-")
            low_v = float(item1_sp[1])
            high_v = float(item1_sp[1])
            if item[0] in nf_norm_dict:
                item_range = nf_norm_dict[item[0]].split("--")
                flag = "阴性"
                if low_v < float(item_range[0]):
                    flag = "阳性"
                if high_v > float(item_range[1]):
                    flag = "阳性"
                nf_res.append(flag)
            else:
                nf_res.append("无法判定")
        else:
            if " mm" in item[1] or "H" in item[1] or "\\" in item[1]:
                item_num = float(item[1].strip(" mH\\."))
            else:
                item_num = float(item[1].replace("..", ".").strip("."))
            if item[0] in nf_norm_dict:
                item_range = nf_norm_dict[item[0]].split("--")
                flag = "阴性"
                if item_num < float(item_range[0]):
                    flag = "阳性"
                if item_num > float(item_range[1]):
                    flag = "阳性"
                nf_res.append(flag)
            else:
                nf_res.append("无法判定")
    else:
        nf_res.append("无法判定")
    return nf_res


def preprocess_labtest_sub_6_5_get_func(line_contend, no_norm_dict):
    # 6.5 for num_other file
    no_res = [line_contend[-1], line_contend[0]]
    chs = "".join(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                   't', 'u', 'v', 'w', 'x', 'y', 'z'])
    chs_up = chs.upper()
    en_neg = {'-', '满意', '低风险', '阴性-', '正常', '基本满意'}
    en_pos = {'+', '+++中量', '++少量', '++++（多量）', '阳性+', '+极少量', '高风险', '阳性'}
    item = line_contend
    if not NlpUtility.has_chinese_char(item[1]):
        if item[1] in en_neg:
            no_res.append("阴性")
        else:
            if item[1] in en_pos:
                no_res.append("阳性")
            else:
                if ":" not in item[1] and "-." not in item[1] and not item[1].startswith("*$*") and item[1] != '.' \
                        and "#" not in item[1] and "（－）" not in item[1] and item[1] != "N" and "nn" not in item[1] \
                        and "：" not in item[1]:
                    if "-" in item[1] or "--" in item[1] or "~" in item[1] or "`" in item[1]:
                        item1_sp = item[1].replace("--", "-").replace("~", "-").replace("`", "-").split("-")
                        low_v = float(item1_sp[1])
                        high_v = float(item1_sp[1])
                        if item[0] in no_norm_dict:
                            item_range = no_norm_dict[item[0]].split("--")
                            flag = "阴性"
                            if low_v < float(item_range[0]):
                                flag = "阳性"
                            if high_v > float(item_range[1]):
                                flag = "阳性"
                            no_res.append(flag)
                        else:
                            no_res.append("无法判定")
                    else:
                        # if "mm" in item[1] or "H" in item[1] or "\\" in item[1] or ">" in item[1] \
                        #         or "<" in item[1] or "%" in item[1] or "＜" in item[1] or "/" in item[1] \
                        #         or "＞" in item[1] or "cm" in item[1] or "H" in item[1] or "CM" in item[1] \
                        #         or "M" in item[1]:
                        #     item_num = float(item[1].strip(" mH\\.<>%＜/＞"+chs+chs_up))
                        # else:
                        #     item_num = float(item[1].replace("..", ".").strip(",."))
                        item_num = float(item[1].replace("..", ".").strip(" mH\\,.<>%＜/＞=" + chs + chs_up))
                        if item[0] in no_norm_dict:
                            item_range = no_norm_dict[item[0]].split("--")
                            flag = "阴性"
                            if item_num < float(item_range[0]):
                                flag = "阳性"
                            if item_num > float(item_range[1]):
                                flag = "阳性"
                            no_res.append(flag)
                        else:
                            no_res.append("无法判定")
                else:
                    no_res.append("无法判定")
    else:
        if item[1] in en_neg:
            no_res.append("阴性")
        else:
            if item[1] in en_pos:
                no_res.append("阳性")
            else:
                no_res.append("无法判定")
    return no_res


def preprocess_labtest_sub_6_6_get_func(line_contend, qc_norm_dict):
    # 6.6 for qual_ch file
    qc_res = [line_contend[-1], line_contend[0]]
    neg_res = {'相符', 'O型', 'B型', 'A型'}
    def __get_ratio_val(val):
        val_sp = val.split(":")
        val_num = ""
        for item in val_sp[1].strip():
            if item in "0123456789.０１２３４５６７８９.":
                val_num += item
            else:
                break
        if len(val_num) == 0:
            return None
        return float(val_num)
    item = line_contend
    item1 = item[1].replace(" ", "")
    if "未" in item1:
        qc_res.append("阴性")
    else:
        if "阴性" in item1 and not item1.endswith("菌"):
            qc_res.append("阴性")
        else:
            if "正常" in item1:
                qc_res.append("阴性")
            else:
                in_neg_dict = False
                for neg_i in neg_res:
                    if neg_i in item1:
                        qc_res.append("阴性")
                        in_neg_dict = True
                        break
                if not in_neg_dict:
                    if "阳性" in item1:
                        qc_res.append("阳性")
                    else:
                        if ":" in item1 or "：" in item1:
                            ratio_val = __get_ratio_val(item1.replace("：", ":"))
                            if ratio_val is None:
                                qc_res.append("无法判定")
                            else:
                                ratio_result = "无法判定"
                                if item[0] in qc_norm_dict:
                                    for norm_item in qc_norm_dict[item[0]]:
                                        if "--" in norm_item:
                                            norm_item_sp = norm_item.split("--")
                                            if ratio_val >= float(norm_item_sp[0]) and ratio_val <= float(norm_item_sp[1]):
                                                ratio_result = "阴性"
                                                break
                                            else:
                                                ratio_result = "阳性"
                                qc_res.append(ratio_result)
                        else:
                            qc_res.append("无法判定")
    return qc_res


def preprocess_labtest_sub_6_7_get_func(line_contend, qn_norm_dict):
    # 6.7 for qual_num file
    qn_res = [line_contend[-1], line_contend[0]]
    chs = "".join(
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
         'v', 'w', 'x', 'y', 'z'])
    chs_up = chs.upper()
    item = line_contend
    if ":" not in item[1] and "-." not in item[1] and not NlpUtility.has_chinese_char(item[1]):
        if "-" in item[1] or ".." in item[1]:
            if "-" in item[1]:
                item1_sp = item[1].split("-")
            else:
                item1_sp = item[1].split("..")
            low_vs = item1_sp[1].replace(" ", "").strip(". %点/日个？"+chs+chs_up)
            low_v = float("-inf") if len(low_vs) == 0 else float(low_vs)
            high_vs = item1_sp[1].replace(" ", "").strip(". %点/日个？"+chs+chs_up)
            high_v = float("inf") if len(high_vs) == 0 else float(high_vs)
            if item[0] in qn_norm_dict:
                item_range = qn_norm_dict[item[0]].split("--")
                flag = "阴性"
                if low_v < float(item_range[0]):
                    flag = "阳性"
                if high_v > float(item_range[1]):
                    flag = "阳性"
                qn_res.append(flag)
            else:
                qn_res.append("无法判定")
        else:
            other_nums = {'1.31*10^5': 131000, '8.52*10^3': 8520, '9.20*10^3': 9200, '2.36*10^3': 2360,
                          '1.93*10^4': 19300, '3.39*10^3': 3390, '4.10*10^3': 4100, '1.90*10^5': 190000,
                          '1.43*10^4': 14300, '7.68*10^3': 7680, '1.19*10^3': 1190, '2.76*10^4':27600, '1.28+E+3':1280}
            garbage_nums = {'0.33E+066.5', '0.8（2）', '5.6.87', '6.01+02', '12.6（3）', '4.70.51', '1.43+03'}
            item1_t = item[1].replace(" ", "").strip(". %点/日个？*"+chs+chs_up).rstrip("+")
            if item1_t in garbage_nums or "（" in item1_t or "）" in item1_t or "#" in item1_t \
                    or item1_t.count(".") >= 2:
                qn_res.append("无法判定")
                return qn_res
            if item1_t in other_nums:
                item_num = other_nums[item1_t]
            else:
                item_num = float(item1_t)
            if item[0] in qn_norm_dict:
                item_range = qn_norm_dict[item[0]].split("--")
                flag = "阴性"
                if item_num < float(item_range[0]):
                    flag = "阳性"
                if item_num > float(item_range[1]):
                    flag = "阳性"
                qn_res.append(flag)
            else:
                qn_res.append("无法判定")
    else:
        qn_res.append("无法判定")
    return qn_res


def preprocess_labtest_sub_6_8_get_func(line_contend, qo_norm_dict):
    # 6.8 for qual_other file
    qo_res = [line_contend[-1], line_contend[0]]
    chs = "".join(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                   't', 'u', 'v', 'w', 'x', 'y', 'z'])
    chs_up = chs.upper()
    qo_neg_val_set = {'--------------', '-----', '----------', '-----------', '-------', '--------', '-------------',
                      '-----.--', '--', '-----.-', '------------', '---------', '------', '----', '---', '-', 'M-',
                      '4-', '-*', 'PCR-', 'neg', 'Neg', 'O', '－', 'norm', '—', 'negative'}
    qo_garbage_val_set = {'M+/-', '+-', '+/-', ':::::', '::::'}
    qo_pos_val_set = {'positive', '（+）', '+++', '1+', '++', '++++', '+++*', '++++*', 'pos', '+++++', 'Pos', '+'}
    qo_val_map = {'?-10.4': "-inf--10.4", '5-----20': "5--20", '5------20': "5--20", '1-5（+）': '1--5',
                  '5----20': "5--20", '1-5+': "1--5", '?-12.7': "-inf--12.7", '０－１': "0--1"}
    num_all_half_map = {"０": "0", "１": "1", "２": "2", "３": "3", "４": "4", "５": "5", "６": "6", "７": "7", "８": "8",
                        "９": "9"}

    def __get_ratio_val(val):
        val_sp = val.split(":")
        val_num = ""
        for item in val_sp[1].strip():
            if item in "0123456789.０１２３４５６７８９.":
                val_num += item
            else:
                break
        if len(val_num) == 0:
            return None
        return float(val_num)
    item = line_contend
    item1 = item[1].replace(" ", "").replace(",", "").replace("。", ".").replace("．", ".").strip()
    for nahm in num_all_half_map.keys():
        item1 = item1.replace(nahm, num_all_half_map[nahm])
    if item1 in qo_neg_val_set:
        qo_res.append("阴性")
    else:
        if item1 in qo_pos_val_set:
            qo_res.append("阳性")
        else:
            if item1 in qo_garbage_val_set:
                qo_res.append("无法判定")
            else:
                if item[0] not in qo_norm_dict:
                    qo_res.append("无法判定")
                else:
                    qo_norm_sp = qo_norm_dict[item[0]].split("--")
                    qo_norm_high_val = float(qo_norm_sp[1])
                    qo_norm_low_val = float(qo_norm_sp[0])

                    daxiao = "<>＜＞》《〈〉"
                    daxiao_res = False
                    for daxiao_i in daxiao:
                        if daxiao_i in item1:
                            daxiao_res = True
                    if daxiao_res:

                        if ":" in item1:
                            qo_real_num = __get_ratio_val(item1)
                        else:
                            cl_item1 = item1.replace("..", ".").strip(" \\,.<>%＜/＞=?\'*+》《〈〉↓" + chs + chs_up)
                            if "+" not in cl_item1 or NlpUtility.has_abc(cl_item1):
                                qo_real_num = float(cl_item1)
                            else:
                                qo_real_num = float(cl_item1.split("+")[0])
                        if ">" in item1 or "＞" in item1 or "》" in item1 or "〉" in item1:
                            if qo_real_num > qo_norm_high_val:
                                qo_res.append("阳性")
                            else:
                                qo_res.append("阴性")
                        else:
                            if qo_real_num < qo_norm_low_val:
                                qo_res.append("阳性")
                            else:
                                qo_res.append("阴性")
                    else:
                        if NlpUtility.is_number(item1):

                            qo_real_num = float(item1)
                            if qo_norm_low_val <= qo_real_num <= qo_norm_high_val:
                                qo_res.append("阴性")
                            else:
                                qo_res.append("阳性")
                        else:
                            if "-" in item1 or "--" in item1 or "~" in item1 or "`" in item1 or "－" in item1:

                                if item1 in qo_val_map:
                                    qo_val_temp_range = qo_val_map[item1].split("--")
                                else:
                                    item1 = item1.replace("－", "-")
                                    qo_val_temp_range = item1.split("-")
                                # print(qo_val_temp_range)
                                try:
                                    qo_val_range_low = float(qo_val_temp_range[0].strip(" \\,.<>%＜/＞=?\'*+》《〈〉↓"))
                                except Exception as e:
                                    qo_val_range_low = 0.0
                                try:
                                    qo_val_range_high = float(qo_val_temp_range[1].strip(" \\,.<>%＜/＞=?\'*+》《〈〉↓"))
                                except Exception as e:
                                    qo_val_range_high = 0.0
                                if qo_val_range_low > qo_norm_high_val:
                                    qo_res.append("阳性")
                                else:
                                    if qo_val_range_high < qo_norm_low_val:
                                        qo_res.append("阴性")
                                    else:
                                        qo_res.append("无法判定")
                            else:
                                if ":" in item1 or "：" in item1:

                                    qo_ratio_real_val = __get_ratio_val(item1.replace("：", ":"))
                                    if qo_norm_low_val <= qo_ratio_real_val <= qo_norm_high_val:
                                        qo_res.append("阴性")
                                    else:
                                        qo_res.append("阳性")
                                else:
                                    if NlpUtility.has_digit(item1):
                                        qo_cl_text = item1.replace("..", ".").strip(
                                            " \\,.<>%＜/＞=?\'*+" + chs + chs_up)
                                        if NlpUtility.is_number(qo_cl_text):
                                            qo_real_num = float(qo_cl_text)
                                            if qo_norm_low_val <= qo_real_num <= qo_norm_high_val:
                                                qo_res.append("阴性")
                                            else:
                                                qo_res.append("阳性")
                                        else:

                                            qo_res.append("无法判定")
                                    else:

                                        qo_res.append("无法判定")
    return qo_res


def free_labtest_preprocess_merge_func(res, **args):
    return __return_res_array(res)


def free_labtest_preprocess_get_func(start, end, **args):
    filename = args['filename']
    split_line_command = args['split_line_command']
    lab_items2kind = args['lab_item2kind']

    nqsc_norm_dict = args['nqsc_norm_dict']
    nqse_norm_dict = args['nqse_norm_dict']
    nf_norm_dict = args['nf_norm_dict']
    no_norm_dict = args['no_norm_dict']
    qn_norm_dict = args['qn_norm_dict']
    qc_norm_dict = args['qc_norm_dict']
    qo_norm_dict = args['qo_norm_dict']

    with open(filename, 'r', encoding='utf-8') as f:
        f.seek(start)
        if start > 0:
            __safe_readline(f)  # drop first incomplete line
        # num_qual_same_ch, num_qual_same_en, num_qual_diff_ch, num_qual_diff_en = [], [], [], []
        # num_float, num_other = [], []
        # qual_float, qual_ch, qual_other = [], [], []
        processed_labtests = []
        line = f.readline()
        while line:
            if split_line_command == "norm":
                split_line = __norm_line_split
            else:  # split_line_command == "sql_command"
                split_line = __conver_line_contend
            contend_list = split_line(line)
            if contend_list is None:
                line = f.readline()
                continue

            ## 处理逻辑 ##
            # 分类讨论不同num qual等不同检验检查取值
            if len(contend_list[2].strip()) > 0:
                if len(contend_list[3].strip()) > 0:
                    if lab_items2kind[contend_list[2].strip()] != lab_items2kind[contend_list[3].strip()]:
                        print("!!! bing cha ji Error : Not Same!!!")
                        labtestID = "none"
                    else:
                        labtestID = lab_items2kind[contend_list[2].strip()]
                else:
                    labtestID = lab_items2kind[contend_list[2].strip()]
            else:
                if len(contend_list[3].strip()) > 0:
                    labtestID = lab_items2kind[contend_list[3].strip()]
                else:
                    print("!!! bing cha ji Error : Empty!!!")
                    labtestID = "none"

            if len(contend_list[5]) > 0:
                if len(contend_list[6]) > 0:
                    if contend_list[5] == contend_list[6]:
                        if NlpUtility.has_chinese_char(contend_list[5]):
                            processed_labtests.append(preprocess_labtest_sub_6_2_get_func([labtestID, contend_list[5], contend_list[6], contend_list[8], contend_list[9],
                                                     contend_list[10], contend_list[12], contend_list[13], contend_list[1]], nqsc_norm_dict))
                        else:
                            processed_labtests.append(preprocess_labtest_sub_6_3_get_func([labtestID, contend_list[5], contend_list[6], contend_list[8], contend_list[9],
                                                     contend_list[10], contend_list[12], contend_list[13], contend_list[1]], nqse_norm_dict))
                    else:
                        if NlpUtility.has_chinese_char(contend_list[6]):
                            processed_labtests.append(preprocess_labtest_sub_6_1_get_func([labtestID, contend_list[5], contend_list[6], contend_list[8], contend_list[9],
                                                     contend_list[10], contend_list[12], contend_list[13], contend_list[1]]))
                        else:
                            processed_labtests.append(preprocess_labtest_sub_6_1_get_func([labtestID, contend_list[5], contend_list[6], contend_list[8], contend_list[9],
                                                     contend_list[10], contend_list[12], contend_list[13], contend_list[1]]))
                else:
                    if NlpUtility.is_number(contend_list[5]):
                        processed_labtests.append(preprocess_labtest_sub_6_4_get_func([labtestID, contend_list[5], contend_list[8], contend_list[9], contend_list[10],
                                          contend_list[12], contend_list[13], contend_list[1]], nf_norm_dict))
                    else:
                        processed_labtests.append(preprocess_labtest_sub_6_5_get_func([labtestID, contend_list[5], contend_list[8], contend_list[9], contend_list[10],
                                          contend_list[12], contend_list[13], contend_list[1]], no_norm_dict))
            else:
                if len(contend_list[6]) > 0:
                    if NlpUtility.is_number(contend_list[6]):
                        processed_labtests.append(preprocess_labtest_sub_6_7_get_func([labtestID, contend_list[6], contend_list[8], contend_list[9], contend_list[10],
                                          contend_list[12], contend_list[13], contend_list[1]], qn_norm_dict))
                    elif NlpUtility.has_chinese_char(contend_list[6]):
                        processed_labtests.append(preprocess_labtest_sub_6_6_get_func([labtestID, contend_list[6], contend_list[8], contend_list[9], contend_list[10],
                                          contend_list[12], contend_list[13], contend_list[1]], qc_norm_dict))
                    else:
                        processed_labtests.append(preprocess_labtest_sub_6_8_get_func([labtestID, contend_list[6], contend_list[8], contend_list[9], contend_list[10],
                                          contend_list[12], contend_list[13], contend_list[1]], qo_norm_dict))

            if f.tell() > end:
                break
            line = f.readline()
    # print(process_index, 'finish~')
    return processed_labtests



def free_labtest56_get_func(start, end, **args):
    filename = args['filename']
    split_line_command = args['split_line_command']
    lab_items2kind = args['lab_item2kind']
    with open(filename, 'r', encoding='utf-8') as f:
        f.seek(start)
        if start > 0:
            __safe_readline(f)  # drop first incomplete line

        num_qual_same_ch, num_qual_same_en, num_qual_diff_ch, num_qual_diff_en = [], [], [], []
        num_float, num_other = [], []
        qual_float, qual_ch, qual_other = [], [], []

        line = f.readline()
        while line:
            if split_line_command == "norm":
                split_line = __norm_line_split
            else:  # split_line_command == "sql_command"
                split_line = __conver_line_contend
            contend_list = split_line(line)
            if contend_list is None:
                line = f.readline()
                continue

            ## 处理逻辑 ##
            # 分类讨论不同num qual等不同检验检查取值
            if len(contend_list[2].strip()) > 0:
                if len(contend_list[3].strip()) > 0:
                    if lab_items2kind[contend_list[2].strip()] != lab_items2kind[contend_list[3].strip()]:
                        print("!!! bing cha ji Error : Not Same!!!")
                        labtestID = "none"
                    else:
                        labtestID = lab_items2kind[contend_list[2].strip()]
                else:
                    labtestID = lab_items2kind[contend_list[2].strip()]
            else:
                if len(contend_list[3].strip()) > 0:
                    labtestID = lab_items2kind[contend_list[3].strip()]
                else:
                    print("!!! bing cha ji Error : Empty!!!")
                    labtestID = "none"

            if len(contend_list[5]) > 0:
                if len(contend_list[6]) > 0:
                    if contend_list[5] == contend_list[6]:
                        if NlpUtility.has_chinese_char(contend_list[5]):
                            num_qual_same_ch.append([labtestID, contend_list[5], contend_list[6], contend_list[8], contend_list[9],
                                                     contend_list[10], contend_list[12], contend_list[13]])
                        else:
                            num_qual_same_en.append([labtestID, contend_list[5], contend_list[6], contend_list[8], contend_list[9],
                                                     contend_list[10], contend_list[12], contend_list[13]])
                    else:
                        if NlpUtility.has_chinese_char(contend_list[6]):
                            num_qual_diff_ch.append([labtestID, contend_list[5], contend_list[6], contend_list[8], contend_list[9],
                                                     contend_list[10], contend_list[12], contend_list[13]])
                        else:
                            num_qual_diff_en.append([labtestID, contend_list[5], contend_list[6], contend_list[8], contend_list[9],
                                                     contend_list[10], contend_list[12], contend_list[13]])
                else:
                    if NlpUtility.is_number(contend_list[5]):
                        num_float.append([labtestID, contend_list[5], contend_list[8], contend_list[9], contend_list[10],
                                          contend_list[12], contend_list[13]])
                    else:
                        num_other.append([labtestID, contend_list[5], contend_list[8], contend_list[9], contend_list[10],
                                          contend_list[12], contend_list[13]])
            else:
                if len(contend_list[6]) > 0:
                    if NlpUtility.is_number(contend_list[6]):
                        qual_float.append([labtestID, contend_list[6], contend_list[8], contend_list[9], contend_list[10],
                                          contend_list[12], contend_list[13]])
                    elif NlpUtility.has_chinese_char(contend_list[6]):
                        qual_ch.append([labtestID, contend_list[6], contend_list[8], contend_list[9], contend_list[10],
                                          contend_list[12], contend_list[13]])
                    else:
                        qual_other.append([labtestID, contend_list[6], contend_list[8], contend_list[9], contend_list[10],
                                          contend_list[12], contend_list[13]])

            if f.tell() > end:
                break
            line = f.readline()
    # print(process_index, 'finish~')
    return num_qual_same_ch, num_qual_same_en, num_qual_diff_ch, num_qual_diff_en, num_float, num_other, qual_float, qual_ch, qual_other


def free_labtest56_merge_func(res, **args):
    num_qual_same_ch, num_qual_same_en, num_qual_diff_ch, num_qual_diff_en, num_float, num_other, qual_float, qual_ch, qual_other = [], [], [], [], [], [], [], [], []
    for i in res:
        num_qual_same_ch.extend(i.get()[0])
        num_qual_same_en.extend(i.get()[1])
        num_qual_diff_ch.extend(i.get()[2])
        num_qual_diff_en.extend(i.get()[3])
        num_float.extend(i.get()[4])
        num_other.extend(i.get()[5])
        qual_float.extend(i.get()[6])
        qual_ch.extend(i.get()[7])
        qual_other.extend(i.get()[8])
    return [num_qual_same_ch, num_qual_same_en, num_qual_diff_ch, num_qual_diff_en, num_float, num_other, qual_float, qual_ch, qual_other]


def get_corpus_contend_thread(start, end, **args):
    filename = args['filename']
    column_set = args['column_set']
    filter_index = [] if args['filter_index'] is None else args['filter_index']
    filter_reverse = args['filter_reverse']
    filter_vals = args['filter_vals']
    cols_num = args['cols_num']
    change_line = args['change_line']
    change_type = args['change_type']
    change_col = args['change_col']
    change_val = args['change_val']
    save_cols = args['save_cols']
    split_line_command = args['split_line_command']
    with open(filename, 'r', encoding='utf-8') as f:
        f.seek(start)
        if start > 0:
            __safe_readline(f)  # drop first incomplete line
        lines = []
        if column_set:
            for set_i in range(len(save_cols)):
                lines.append(set([]))
        line = f.readline()
        while line:
            if split_line_command == "norm":
                split_line = __norm_line_split
            else:  # split_line_command == "sql_command"
                split_line = __conver_line_contend
            contend_list = split_line(line)
            if contend_list is None:
                line = f.readline()
                continue

            #  数据处理逻辑  ##
            cond_flag = not filter_reverse
            for filter_iter, filter_item in enumerate(filter_index):
                # print(filter_reverse, filter_item, filter_iter)
                if (not filter_reverse and contend_list[filter_item] not in filter_vals[filter_iter]) or \
                        (filter_reverse and contend_list[filter_item] not in filter_vals[filter_iter]):
                    cond_flag = filter_reverse
                if cond_flag == filter_reverse:
                    break
            if cond_flag:
                if change_line:
                    contend_list = __change_line(contend_list, change_type, change_col, change_val)
                if len(contend_list) != cols_num:
                    print(contend_list)
                else:
                    if column_set:
                        for save_set_i in range(len(save_cols)):
                            lines[save_set_i].add(contend_list[save_cols[save_set_i]])
                    else:
                        lines.append([contend_list[i] for i in save_cols])
            elif change_line:
                if len(contend_list) != cols_num:
                    print(contend_list)
                else:
                    if column_set:
                        for save_set_i in range(len(save_cols)):
                            lines[save_set_i].add(contend_list[save_cols[save_set_i]])
                    else:
                        lines.append([contend_list[i] for i in save_cols])

            if f.tell() > end:
                break
            line = f.readline()
    # print(process_index, 'finish~')
    return lines


def merge_corpus_contend_thread(res, **args):
    column_set = args['column_set']
    if column_set:
        print(len(res))
        return __return_res_set(res)
    else:
        return __return_res_array(res)


def __return_res_array(res):
    filted_contend = []
    for i in res:
        filted_contend.extend(i.get())
    return filted_contend


def __return_res_set(res):
    filted_contend = []
    for i_iter, i in enumerate(res):
        i_sets = i.get()
        if i_iter == 0:
            for i_s_iter, i_s in enumerate(i_sets):
                filted_contend.append(i_s)
        else:
            for i_s_iter, i_s in enumerate(i_sets):
                filted_contend[i_s_iter].update(i_s)
    return filted_contend


def load_and_process_bigdata(ipl_get_func=get_corpus_contend_thread, ipl_merge_func=merge_corpus_contend_thread, **ipl_args):
    # 自由查询需要自己实现处理函数 free_ipl_fuc
    # 模式查询分成两个模式：
    # 1. column_set为False：按照指定值filter_val，过滤指定列filter_vals，按照save_cols存储原始值，split_line代表每一行的分割函数
    # 2. column_set为True：按照指定值filter_val，过滤指定列filter_vals，按照save_cols存储指定列的set值集合，split_line代表每一行的分割函数
    # 修改分为三种模式：（设定change_line为True）
    # 1. 增加，设定change_type为"INSERT"，在change_col位置插入change_val值
    # 2. 删除，设定change_type为"DELETE"，在change_col位置删除原有值
    # 3. 删除空余值，设定change_type为"DEL_EMP"，所有位置删除空值
    # 4. 修改，设定change_type为"CHANGE"，在change_col位置将原有值修改为change_val值
    # 5. 条件修改，设定change_type为"COND_CHANGE"，在change_col位置，按照不同的dict条件：满足key的条件，更改为value值

    filename = ipl_args['filename']
    num_processor = ipl_args['num_processor']
    begin_time = time()
    # Break the files into num_threads batches.
    with open(filename, 'r', encoding='utf-8') as f:
        size = os.fstat(f.fileno()).st_size  # 指针操作，所以无视文件大小
    spacing = np.linspace(0, size, num_processor + 1).astype(np.int64)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    p = Pool(num_processor)
    res = []
    # print(ipl_args, type(ipl_args))
    for i in range(num_processor):
        start = ranges[i][0]
        end = ranges[i][1]
        # print(start, end)
        res.append(p.apply_async(ipl_get_func, (start, end), ipl_args))
        # print(str(i) + ' processor started !')
    p.close()
    p.join()
    end_time = time()
    print("consume time:", end_time - begin_time)
    return ipl_merge_func(res, **ipl_args)


def get_string_match_score(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


def match_concept_get_func(start, end, **args):
    to_split_list = args['to_split_list']
    origin_kg = args['origin_kg']
    base_sim = args['base_sim']

    diagnose_match_dict = {}
    diagnose_miss = 0
    for item in to_split_list[start: end]:
        item_score = 0
        item_match = None
        for kg_concept in origin_kg.keys():
            temp_match_score = get_string_match_score(item, kg_concept)
            if temp_match_score < base_sim:
                temp_match_score = 0
            if temp_match_score > item_score:
                item_score = temp_match_score
                item_match = kg_concept
        if item_score == 0:
            diagnose_miss += 1
        else:
            diagnose_match_dict[item] = item_match
    return diagnose_miss, diagnose_match_dict


def match_labtest_concept_get_func(start, end, **args):
    to_split_list = args['to_split_list']
    origin_kg = args['origin_kg']
    base_sim = args['base_sim']
    lab_kind2items = args['lab_kind2items']

    diagnose_match_dict = {}
    diagnose_miss = 0
    for item in to_split_list[start: end]:
        item_score = 0
        item_match = None
        for item_name in lab_kind2items[item]:
            for kg_concept in origin_kg.keys():
                temp_match_score = get_string_match_score(item_name, kg_concept)
                if temp_match_score < base_sim:
                    temp_match_score = 0
                if temp_match_score > item_score:
                    item_score = temp_match_score
                    item_match = kg_concept
        if item_score == 0:
            diagnose_miss += 1
        else:
            diagnose_match_dict[item] = item_match
    return diagnose_miss, diagnose_match_dict


def match_concept_merge_func(res, **args):
    diagnose_sum_miss = 0
    diagnose_sum_dict = {}
    for i in res:
        diagnose_sum_miss += i.get()[0]
        diagnose_sum_dict.update(i.get()[1])
    return diagnose_sum_miss, diagnose_sum_dict


def process_bigdata(ipl_get_func, ipl_merge_func, **ipl_args):

    num_processor = ipl_args['num_processor']
    to_split_list = ipl_args['to_split_list']
    begin_time = time()
    # Break the files into num_threads batches.

    size = len(to_split_list)
    spacing = np.linspace(0, size, num_processor + 1).astype(np.int64)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    p = Pool(num_processor)
    res = []
    # print(ipl_args, type(ipl_args))
    for i in range(num_processor):
        start = ranges[i][0]
        end = ranges[i][1]
        # print(start, end)
        res.append(p.apply_async(ipl_get_func, (start, end), ipl_args))
        # print(str(i) + ' processor started !')
    p.close()
    p.join()
    end_time = time()
    print("consume time:", end_time - begin_time)
    return ipl_merge_func(res, **ipl_args)

# #------------------------- test ----------------------#
# if __name__ == '__main__':
#     test2(p="fwefwef", sd="cwefwe")
