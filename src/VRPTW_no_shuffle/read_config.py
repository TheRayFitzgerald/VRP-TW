

def read_config():
	vars_dict = {"MAX_NUMBER_OF_ROUTES":0, "SPEED":0, "NUMBER_OF_ORDERS":0, "CONVERGENCE_COUNTER":0}

	for line in open('config.txt', 'r').readlines():
		
		vars_dict[line.split()[0]] = int(line.split()[1])

	return vars_dict

if __name__ == '__main__':
	read_config()	