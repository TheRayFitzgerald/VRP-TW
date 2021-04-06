import json, statistics

ROUND = 2

with open('results_3.json') as file:
    results = json.load(file)

print(results)

alg_1_runtimes = [runtime_pair[0] for runtime_pair in results['runtimes']]
alg_2_runtimes = [runtime_pair[1] for runtime_pair in results['runtimes']]

alg_1_distances = [distance_pair[0] for distance_pair in results['distances']]
alg_2_distances = [distance_pair[1] for distance_pair in results['distances']]

mean_runtimes = (round(statistics.mean(alg_1_runtimes), ROUND), round(statistics.mean(alg_2_runtimes), ROUND))
stdev_runtimes = (round(statistics.stdev(alg_1_runtimes), ROUND), round(statistics.stdev(alg_2_runtimes), ROUND))

mean_distances = (round(statistics.mean(alg_1_distances), ROUND), round(statistics.mean(alg_2_distances), ROUND))



#mean_runtimes = [round(total_runtime_1/results['config']['N_SAMPLES'], 2), round(total_runtime_2/results['config']['N_SAMPLES'], 2)]

print(mean_runtimes)
print(mean_distances)
print()
#print(stdev_runtimes)
    