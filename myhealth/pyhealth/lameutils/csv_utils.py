# -*- coding:utf-8 -*-
import pickle
class CsvUtility(object):
	@staticmethod
	def read_norm_array_csv(file_name, sp="\t"):
		result = []
		with open(file_name, 'r', encoding='utf-8') as f:
			for row in f.readlines():  # 将csv 文件中的数据保存到birth_data中
				result.append(row.strip().split(sp))
		return result

	@staticmethod
	def reform_word_dict(file_name):
		raw_data = CsvUtility.read_norm_array_csv(file_name)
		reform_data = []
		for line in range(1, len(raw_data)):
			sp_line = raw_data[line].split(",")
			print(sp_line)
			reform_data.append([",".join(sp_line[:-1]), sp_line[-1]])
		return reform_data

	@staticmethod
	def write_word_dict(file_name, data):
		with open(file_name, 'w', encoding='utf-8') as f:
			for idata in data.items():
				f.write(str(idata[0]) + "\t" + str(idata[1]) + "\n")

	@staticmethod
	def write_set(file_name, data):
		with open(file_name, 'w', encoding='utf-8') as f:
			for idata in data:
				f.write(str(idata) + "\n")


	@staticmethod
	def read_word_dict(file_name, sp="\t"):
		res_dict = {}
		rep_count = 0
		with open(file_name, 'r', encoding='utf-8')as f:
			for row in f.readlines():
				row_sp = row.split(sp)
				if len(row_sp) != 2:
					print("wrong line: ", row_sp, "length: ", len(row_sp))
				if row_sp[0] in res_dict:
					rep_count += 1
				res_dict[row_sp[0]] = row_sp[1].strip()
		return res_dict, rep_count


	@staticmethod
	def write_relation(file_name, map_key_data, map_value_data):
		with open(file_name, 'w', encoding='utf-8') as f:
			for id, key in enumerate(map_key_data):
				for value in map_value_data[id]:
					f.write(key + '\t' + value + '\n')

	@staticmethod
	def write_3tuple(file_name, head_data, rel_data, end_data, w_t='w'):
		assert len(head_data) == len(rel_data) and len(rel_data) == len(end_data)
		with open(file_name, w_t, encoding='utf-8') as f:
			for id, key in enumerate(head_data):
				f.write(key + '\t' + rel_data[id] + '\t' + end_data[id] + '\n')

	@staticmethod
	def write_norm_array_csv(file_name, data, sp='\t', w_t='w'):
		with open(file_name, w_t, encoding='utf-8') as f:
			for idata in data:
				f.write(sp.join(idata) + "\n")

	@staticmethod
	def read_top_lines(file_name, top_k):
		result = []
		with open(file_name, 'r', encoding='utf-8') as f:
			for i in range(top_k):
				result.append(f.readline())
		return result

	@staticmethod
	def write_pickle(save_path, data, wt='wb'):
		with open(save_path, wt) as f:
			pickle.dump(data, f)

	@staticmethod
	def read_pickle(save_path, rt='rb'):
		with open(save_path, rt) as f:
			data = pickle.load(f)
		return data
