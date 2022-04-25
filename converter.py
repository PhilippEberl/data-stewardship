from sklearn.impute import SimpleImputer
import numpy as np
import re
import datetime as dt

def convert_to_numeric(vocabulary, line_arr, ignore=[], nanval=""):
	ret = []

	for i, attr in enumerate(line_arr):
		if i in ignore:
			ret.append(attr)
			continue

		if attr.replace(".","").isnumeric():
			if "." in attr:
				ret.append(float(attr))
			else:
				ret.append(int(attr))
			continue

		if i not in vocabulary:
			vocabulary[i] = {}
			if nanval != "":
				vocabulary[i][nanval] = -1

		if attr not in vocabulary[i]:
			vocabulary[i][attr] = int(list(vocabulary[i].values())[-1] + 1 if len(vocabulary[i]) > 0 else 0)

		ret.append(vocabulary[i][attr])

	return ret

def one_to_n(data, columns):
	# find number of elements per column
	elnum = [0] * len(columns)
	for line in data:
		for i in range(len(columns)):
			elnum[i] = max(line[columns[i]]+1, elnum[i])

	# convert columns
	ret = []
	for line in data:
		prev = 0
		arr = []
		for i in range(len(columns)):
			conv = [0] * elnum[i]
			conv[line[columns[i]]] = 1

			arr += list(line[prev:columns[i]]) + conv
			prev = columns[i] + 1
		arr += list(line[prev:])
		ret.append(arr)

	return ret

def convert_date_to_unix_time(data, columns):
	epoch = dt.datetime.utcfromtimestamp(0)

	ret = []
	for line in data:
		prev = 0
		arr = []
		for i in range(len(columns)):
			conv = int((dt.datetime.strptime(line[columns[i]], "%Y-%m-%d") - epoch).total_seconds() * 1000.0)

			arr += list(line[prev:columns[i]]) + [conv]
			prev = columns[i] + 1
		arr += list(line[prev:])
		ret.append(arr)

	return ret

def impute_values(data):
	imp = SimpleImputer(missing_values=-1, strategy='most_frequent')
	return imp.fit_transform(data)

def join_array(array, seperator=";"):
	if len(array) > 0:
		ret = str(array[0])
	else:
		return ""

	for i in array[1:]:
		ret += ";" + str(i)

	return ret

def convert_ecoli():
	data = []
	vocabulary = {}

	# open file
	with open("../datasets/ecoli/ecoli.data", "r") as f:
		fl = True
		for l in f:
			if fl: fl = False; continue
			line_arr = re.sub(r" +", " ", l.strip()).split(" ")
			line_arr = convert_to_numeric(vocabulary, line_arr)
			data.append(line_arr)

	# save converted
	with open("../datasets/ecoli/ecoli_conv.data", "w") as f:
		for d in data:
			f.write(join_array(d[1:]) + "\n")

	# save input data
	with open("../datasets/ecoli/ecoli_input.data", "w") as f:
		for d in data:
			f.write(join_array(d[1:-1]) + "\n")

	# save class ids
	with open("../datasets/ecoli/ecoli_classes.data", "w") as f:
		for d in data:
			f.write(str(d[-1]) + "\n")

def convert_mushrooms():
	data = []
	vocabulary = {}

	# open file
	with open("../datasets/mushrooms/agaricus-lepiota.data", "r") as f:
		for l in f:
			line_arr = l.strip().split(",")
			line_arr = convert_to_numeric(vocabulary, line_arr, nanval="?")
			data.append(line_arr)

	data = impute_values(data)
	data = one_to_n(data, [1, 2, 3, 5, 6, 7, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22])

	# save converted
	with open("../datasets/mushrooms/agaricus-lepiota_conv.data", "w") as f:
		for d in data:
			f.write(join_array(d) + "\n")

	# save input data
	with open("../datasets/mushrooms/agaricus-lepiota_input.data", "w") as f:
		for d in data:
			f.write(join_array(d[1:]) + "\n")

	# save class ids
	with open("../datasets/mushrooms/agaricus-lepiota_classes.data", "w") as f:
		for d in data:
			f.write(str(d[0]) + "\n")

def convert_weather():
	data = []
	vocabulary = {}

	# open file
	with open("../datasets/rain/weatherAUS.csv", "r") as f:
		fl = True
		for l in f:
			if fl: fl = False; continue
			line_arr = l.strip().split(",")
			line_arr = convert_to_numeric(vocabulary, line_arr)
			data.append(line_arr)

	# save converted
	with open("../datasets/rain/weatherAUS_conv.csv", "w") as f:
		for d in data:
			f.write(";".join(d[1:]) + "\n")

	# save input data
	with open("../datasets/rain/weatherAUS_input.data", "w") as f:
		for d in data:
			f.write(";".join(d[1:-1]) + "\n")

	# save class ids
	with open("../datasets/rain/weatherAUS_classes.data", "w") as f:
		for d in data:
			f.write(d[-1] + "\n")

def convert_music():
	data = []
	vocabulary = {}

	# open file
	with open("../datasets/music/train.csv", "r", encoding="utf-8") as f:
		fl = True
		for l in f:
			if fl: fl = False; continue
			line_arr = l.strip().split(",")
			line_arr = convert_to_numeric(vocabulary, line_arr[-14:])
			data.append(line_arr)

	# save converted
	with open("../datasets/music/train_conv.csv", "w") as f:
		for d in data:
			if (len(d[2:]) == 16):
				f.write(";".join(d[3:]) + "\n")
			else:
				f.write(";".join(d[2:]) + "\n")

	# save input data
	with open("../datasets/music/train_input.data", "w") as f:
		for d in data:
			if (len(d[2:-1]) == 15):
				f.write(";".join(d[3:-1]) + "\n")
			else:
				f.write(";".join(d[2:-1]) + "\n")

	# save class ids
	with open("../datasets/music/train_classes.data", "w") as f:
		for d in data:
			f.write(d[-1] + "\n")

def convert_congress():
	data = []
	vocabulary = {}

	# open file
	with open("../datasets/congress/CongressionalVotingID.shuf.lrn.csv", "r", encoding="utf-8") as f:
		fl = True
		for l in f:
			if fl: fl = False; continue
			line_arr = l.strip().split(",")
			line_arr = convert_to_numeric(vocabulary, line_arr[1:], nanval="unknown")
			data.append(line_arr)

	data = impute_values(data)

	# save converted
	with open("../datasets/congress/CongressionalVotingID_conv.shuf.lrn.data", "w") as f:
		for d in data:
			f.write(join_array(d[1:]) + ";" + str(d[0]) + "\n")


	# save input data
	with open("../datasets/congress/CongressionalVotingID_input.shuf.lrn.data", "w") as f:
		for d in data:
			f.write(join_array(d[1:]) + "\n")

	# save class ids
	with open("../datasets/congress/CongressionalVotingID_classes.shuf.lrn.data", "w") as f:
		for d in data:
			f.write(str(d[0]) + "\n")

def convert_location():
	data = []
	vocabulary = {}

	# open file
	with open("../datasets/location/Location446-30cls-5k.lrn.csv", "r", encoding="utf-8") as f:
		fl = True
		for l in f:
			if fl: fl = False; continue
			line_arr = l.strip().split(",")
			line_arr = convert_to_numeric(vocabulary, line_arr[1:])
			data.append(line_arr)

	# save converted
	with open("../datasets/location/Location446-30cls-5k_conv.lrn.data", "w") as f:
		for d in data:
			f.write(join_array(d[:-1]) + ";" + str(d[-1]) + "\n")


	# save input data
	with open("../datasets/location/Location446-30cls-5k_input.lrn.data", "w") as f:
		for d in data:
			f.write(join_array(d[:-1]) + "\n")

	# save class ids
	with open("../datasets/location/Location446-30cls-5k_classes.lrn.data", "w") as f:
		for d in data:
			f.write(str(d[-1]) + "\n")

def convert_ozone():
	data = []
	vocabulary = {}

	# open file
	with open("../datasets/ozone/eighthr.data", "r") as f:
		for l in f:
			line_arr = l.strip().split(",")
			line_arr = convert_to_numeric(vocabulary, line_arr[1:], nanval="?")
			data.append(line_arr)

	data = impute_values(data)

	# save converted
	with open("../datasets/ozone/eighthr_conv.data", "w") as f:
		for d in data:
			f.write(join_array(d[1:]) + "\n")

	# save input data
	with open("../datasets/ozone/eighthr_input.data", "w") as f:
		for d in data:
			f.write(join_array(d[1:-1]) + "\n")

	# save class ids
	with open("../datasets/ozone/eighthr_classes.data", "w") as f:
		for d in data:
			f.write(str(d[-1]) + "\n")

def convert_kaggle():
	data = []
	ids = []
	vocabulary = {}

	# open file
	with open("../datasets/location/Location446-30cls-5k.tes.csv", "r") as f:
		fl = True
		for l in f:
			if fl: fl = False; continue
			line_arr = l.strip().split(",")
			ids.append(line_arr[0])
			line_arr = convert_to_numeric(vocabulary, line_arr[1:], nanval="?")
			data.append(line_arr)

	# save input data
	with open("../datasets/location/competition_input.data", "w") as f:
		for d in data:
			f.write(join_array(d) + "\n")

	# save ids data
	with open("../datasets/location/competition_ids.data", "w") as f:
		for d in ids:
			f.write(d + "\n")

	data = []
	ids = []
	vocabulary = {}

	# open file
	with open("../datasets/congress/CongressionalVotingID.shuf.tes.csv", "r") as f:
		fl = True
		for l in f:
			if fl: fl = False; continue
			line_arr = l.strip().split(",")
			ids.append(line_arr[0])
			line_arr = convert_to_numeric(vocabulary, line_arr[1:], nanval="unknown")
			data.append(line_arr)


	data = impute_values(data)

	# save input data
	with open("../datasets/congress/competition_input.data", "w") as f:
		for d in data:
			f.write(join_array(d) + "\n")

	# save ids data
	with open("../datasets/congress/competition_ids.data", "w") as f:
		for d in ids:
			f.write(d + "\n")

def convert_solar_flairs():
	data = []
	vocabulary = {}

	# open file
	with open("../datasets/exercise2/solarflares/flare.data", "r") as f:
		fl = True
		for l in f:
			if fl: fl = False; continue
			line_arr = l.strip().split(" ")
			line_arr = convert_to_numeric(vocabulary, line_arr, nanval="")
			data.append(line_arr)

	data = one_to_n(data, [0,1,2])

	# save converted
	with open("../datasets/exercise2/solarflares/flare_conv.data", "w") as f:
		for d in data:
			f.write(join_array(d) + "\n")

	# save input data
	with open("../datasets/exercise2/solarflares/flare_input.data", "w") as f:
		for d in data:
			f.write(join_array(d[0:-3]) + "\n")

	# save class ids
	with open("../datasets/exercise2/solarflares/flare_classes.data", "w") as f:
		for d in data:
			f.write(join_array(d[-3:]) + "\n")

def convert_wine():
	for s in ["red", "white"]:
		data = []
		vocabulary = {}

		# open file
		with open("../datasets/exercise2/wine/winequality-" + s + ".csv", "r") as f:
			fl = True
			for l in f:
				if fl: fl = False; continue
				line_arr = l.strip().split(";")
				line_arr = convert_to_numeric(vocabulary, line_arr, nanval="")
				data.append(line_arr)

		# save converted
		with open("../datasets/exercise2/wine/wine_" + s + "_conv.data", "w") as f:
			for d in data:
				f.write(join_array(d) + "\n")

		# save input data
		with open("../datasets/exercise2/wine/wine_" + s + "_input.data", "w") as f:
			for d in data:
				f.write(join_array(d[0:-1]) + "\n")

		# save class ids
		with open("../datasets/exercise2/wine/wine_" + s + "_classes.data", "w") as f:
			for d in data:
				f.write(str(d[-1]) + "\n")

def convert_covid():
	data = []
	vocabulary = {}

	# open file
	with open("../datasets/exercise2/covid/covid-vaccination-vs-death_ratio.csv", "r") as f:
		fl = True
		for l in f:
			if fl: fl = False; continue
			line_arr = l.strip().replace(", ", " ").split(",")
			line_arr = convert_to_numeric(vocabulary, line_arr[1:], ignore=[2], nanval="")
			data.append(line_arr)

	data = convert_date_to_unix_time(data, [2])

	for line in data:
		line[3] = line[3]/line[7]
		line[4] = line[4]/line[7]
		line[5] = line[5]/line[7]
		line[6] = line[6]/line[7]


	data = one_to_n(data, [0,1])

	# save converted
	with open("../datasets/exercise2/covid/covid-vaccination-vs-death_ratio_conv.data", "w") as f:
		for d in data:
			f.write(join_array(d) + "\n")

	# save input data
	with open("../datasets/exercise2/covid/covid-vaccination-vs-death_ratio_input.data", "w") as f:
		for d in data:
			f.write(join_array(d[0:-1]) + "\n")

	# save class ids
	with open("../datasets/exercise2/covid/covid-vaccination-vs-death_ratio_classes.data", "w") as f:
		for d in data:
			f.write(str(d[-1]) + "\n")

def main():
	#convert_ecoli()
	#print("converted ecoli dataset")
	#convert_mushrooms()
	#print("converted mushrooms dataset")
	#convert_weather()
	#print("converted weather dataset")
	#convert_music()
	#print("converted music dataset")
	#convert_congress()
	#print("converted congress dataset")
	#convert_location()
	#print("converted location dataset")
	#convert_kaggle()
	#print("converted kaggle dataset")

	# exercise 3
	#convert_solar_flairs()
	#print("converted solar flares dataset")
	#convert_wine()
	#print("converted wine dataset")
	convert_covid()
	print("converted covid dataset")

if __name__ == "__main__":
	main()