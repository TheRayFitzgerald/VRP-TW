import json, statistics

ROUND = 2

with open('results_7.json') as file:
    results = json.load(file)

print(results)

def main1(results):

	mean_runtimes = round(statistics.mean(results['runtimes']), ROUND)
	mean_distances = round(statistics.mean(results['distances']), ROUND)
	mean_n_routes = round(statistics.mean(results['n_routes']), ROUND)

	print('########')
	#print(mean_runtimes)
	#print(mean_distances)
	print(mean_n_routes)
	print('########\n\n')

def main2(results):
	alg_1_runtimes = [runtime_pair[0] for runtime_pair in results['runtimes']]
	alg_2_runtimes = [runtime_pair[1] for runtime_pair in results['runtimes']]

	alg_1_distances = [distance_pair[0] for distance_pair in results['distances']]
	alg_2_distances = [distance_pair[1] for distance_pair in results['distances']]

	#alg_1_num_seed_routes = [num_seed_routes_pair[0] for num_seed_routes_pair in results['num_seed_routes']]
	#alg_2_num_seed_routes = [num_seed_routes_pair[1] for num_seed_routes_pair in results['num_seed_routes']]

	#alg_1_num_routes = [num_routes_pair[0] for num_routes_pair in results['total_num_routes']]
	#alg_2_num_routes = [num_routes_pair[1] for num_routes_pair in results['total_num_routes']]

	mean_runtimes = (round(statistics.mean(alg_1_runtimes), ROUND), round(statistics.mean(alg_2_runtimes), ROUND))
	stdev_runtimes = (round(statistics.stdev(alg_1_runtimes), ROUND), round(statistics.stdev(alg_2_runtimes), ROUND))

	mean_distances = (round(statistics.mean(alg_1_distances), ROUND), round(statistics.mean(alg_2_distances), ROUND))

	#total_num_seed_routes = (sum(alg_1_num_seed_routes), sum(alg_2_num_seed_routes))
	#total_num_routes = (sum(alg_1_num_routes), sum(alg_2_num_routes))

	#mean_runtimes = [round(total_runtime_1/results['config']['N_SAMPLES'], 2), round(total_runtime_2/results['config']['N_SAMPLES'], 2)]

	print(mean_runtimes)
	print(mean_distances)
	#print(total_num_seed_routes)
	#print(total_num_routes)
	#print(stdev_runtimes)




def main3(results):

	alg_1_runtimes = [runtime_pair[0] for runtime_pair in results['runtimes']]
	alg_2_runtimes = [runtime_pair[1] for runtime_pair in results['runtimes']]
	alg_3_runtimes = [runtime_pair[2] for runtime_pair in results['runtimes']]

	alg_1_distances = [distance_pair[0] for distance_pair in results['distances']]
	alg_2_distances = [distance_pair[1] for distance_pair in results['distances']]
	alg_3_distances = [distance_pair[2] for distance_pair in results['distances']]


	mean_runtimes = [round(statistics.mean(alg_1_runtimes), ROUND), round(statistics.mean(alg_2_runtimes), ROUND), round(statistics.mean(alg_3_runtimes), ROUND)]
	mean_distances = [round(statistics.mean(alg_1_distances), ROUND), round(statistics.mean(alg_2_distances), ROUND), round(statistics.mean(alg_3_distances), ROUND)]

	print(mean_runtimes)
	print(mean_distances)

if __name__ == '__main__':
	#main3(results)
	
	
	for result_set_key in results.keys():
		if result_set_key != 'config':
			print(result_set_key)
			main1(results[result_set_key])

    