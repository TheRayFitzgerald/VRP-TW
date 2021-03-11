

def read_config():
	print('here')
	vars_dict = {"MAX_NUMBER_OF_ROUTES":0, "SPEED":0, "NUMBER_OF_ORDERS":0, "CONVERGENCE_COUNTER":0}

	for line in open('config.txt', 'r').readlines():
		#key_value = line.split()
		print(line.split())
		vars_dict[line.split()[0]] = int(line.split()[1])

	print(vars_dict)
	return vars_dict

if __name__ == '__main__':
	read_config()	