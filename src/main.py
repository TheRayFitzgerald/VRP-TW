from math import sqrt
from random import random, randrange
from operator import itemgetter, attrgetter
from GRASP1_oldcost import grasp as grasp2, Order, plot_routes, routes_are_feasible, get_overall_distance
from GRASP1 import grasp as grasp1
from read_config import read_config
import matplotlib.pyplot as plt
import random, pickle, datetime, time, os, sys, json, copy
#import numpy as np

results = dict()

results['runtimes'] = list()
results['distances'] = list()
#results['num_seed_routes'] = list()
#results['total_num_routes'] = list()
#results['winners'] = list()

results['config'] = dict()

config_vars = read_config()
for key,val in config_vars.items():
    exec(key + '=val')
    results['config'][key] = val

#GRAPH_ROUTES = bool(GRAPH_ROUTES)

# write out the 0% initial progress bar
i=0
sys.stdout.write('\r')
sys.stdout.write("[%-10s] %d%%" % ('='*10*i, 100*i/N_SAMPLES))
sys.stdout.flush()

def main():

    #create a list of orders
    for i in range(N_SAMPLES):

        blockPrint()

        orders_1 = create_orders(NUMBER_OF_ORDERS)
        orders_2 = copy.deepcopy(orders_1)

        start = time.time()
        routes_1 = grasp1(orders_1, False)
        routes_1_time = round(time.time() - start, 3)
        routes_1_distance = round(get_overall_distance(routes_1), 3)

        start = time.time()
        routes_2 = grasp2(orders_2, False)
        routes_2_time = round(time.time() - start, 3)
        routes_2_distance = round(get_overall_distance(routes_2), 3)
        
        results['runtimes'].append([routes_1_time, routes_2_time])
        results['distances'].append([routes_1_distance, routes_2_distance])

        enablePrint()

        sys.stdout.write('\r')
        sys.stdout.write("[%-30s] %d%%" % ('='*10*(i+1), 100*(i+1)/N_SAMPLES))
        sys.stdout.flush()
        time.sleep(0.25)

        '''
        if routes_1_time < routes_2_time:
            results['winners'].append(1)
        elif routes_1_time > routes_2_time:
            results['winners'].append(2)
        else:
            results['winners'].append(0)
        '''

    enablePrint()
    # print a new line
    print()

    with open('results.json', 'w') as f:
        json.dump(results, f)


    if GRAPH_ROUTES:
        plot_routes(routes_1, "grasp1")
        plot_routes(routes_2, "grasp2") 
    '''
    print(routes_are_feasible(routes_1))
    print(get_overall_distance(routes_1))
    print(routes_are_feasible(routes_2))
    print(get_overall_distance(routes_2))

    print('Times')
    print(routes_1_time)
    print(routes_2_time)
    '''
    if GRAPH_ROUTES:
        plt.show()


    print('Done')
    print(results)
    return routes_are_feasible(routes_2)

    
    if not routes:
        a = input("Main Routing Fail. Would you like to save these Orders? [y/n]: ")
        if a == 'y':
            i = 0
            while os.path.exists("../saved_orders/orders_%s.pkl" % i):
                i += 1
            with open("../saved_orders/orders_%s.pkl" % i, 'wb') as output:
                pickle.dump(orders, output, pickle.HIGHEST_PROTOCOL)
                print('Saved')
        else:
            print('Exiting')
    
    else:
        a = input("Would you like to save these Routes and Orders? [y/n]: ")
        if a == 'y':
            i = 0
            while os.path.exists("../saved_routes/routes_%s.pkl" % i):
                i += 1
            filename = "../saved_routes/routes_%s.pkl" % i
            with open(filename, 'wb') as output:
                pickle.dump(routes, output, pickle.HIGHEST_PROTOCOL)
                print('Saved routes to %s' % filename)

            i = 0
            while os.path.exists("../saved_orders/orders_%s.pkl" % i):
                i += 1
            filename = "../saved_orders/orders_%s.pkl" % i
            with open(filename, 'wb') as output:
                pickle.dump(orders, output, pickle.HIGHEST_PROTOCOL)
                print('Saved orders to %s' % filename)

            
        else:
            print('Exiting')


def main2():
   
    with open('../saved_orders/orders_10.pkl', 'rb') as input:
        orders = pickle.load(input)

    routes = grasp(orders, True)
    if not routes:
        a = input("Main Routing Fail. Would you like to save these Orders? [y/n]: ")
        if a == 'y':
            i = 0
            while os.path.exists("../saved_orders/orders_%s.pkl" % i):
                i += 1
            with open("../saved_orders/orders_%s.pkl" % i, 'wb') as output:
                pickle.dump(orders, output, pickle.HIGHEST_PROTOCOL)
                print('Saved')
        else:
            print('Exiting')
    
    else:
        a = input("Would you like to save these Routes? [y/n]: ")
        if a == 'y':
            i = 0
            while os.path.exists("../saved_routes/routes_%s.pkl" % i):
                i += 1
            filename = "../saved_routes/routes_%s.pkl" % i
            with open(filename, 'wb') as output:
                pickle.dump(routes, output, pickle.HIGHEST_PROTOCOL)
                print('Saved to %s' % filename)
        else:
            print('Exiting')

def main3():

    blockPrint()

    orders = create_orders(NUMBER_OF_ORDERS)

    start = time.time()
    routes_1 = grasp2(orders, True)
    routes_1_time = round(time.time() - start, 3)
    routes_1_distance = round(get_overall_distance(routes_1), 3)

    enablePrint()         

    sys.stdout.write('\r')
    sys.stdout.write("[%-10s] %d%%" % ('='*10*(i+1), 100*(i+1)/N_SAMPLES))
    sys.stdout.flush()
    time.sleep(0.25)
    print()
    print('Total Runtime: %.3f Seconds' % routes_1_time)
    print('Total Distance: %.3f Meters' % routes_1_distance)

    blockPrint()

    if GRAPH_ROUTES:
        plot_routes(routes_1, "Final Routes")
        plt.show()

def main4():

    #create a list of orders
    for i in range(N_SAMPLES):

        blockPrint()

        orders_1 = create_orders(NUMBER_OF_ORDERS)
        orders_2 = copy.deepcopy(orders_1)

        start = time.time()
        routes_1 = grasp1(orders_1, False)
        routes_1_time = round(time.time() - start, 3)
        routes_1_distance = round(get_overall_distance(routes_1), 3)
  
        start = time.time()
        routes_2 = grasp2(orders_2, False)
        routes_2_time = round(time.time() - start, 3)
        routes_2_distance = round(get_overall_distance(routes_2), 3)
       
        results['runtimes'].append([routes_1_time, routes_2_time])
        results['distances'].append([routes_1_distance, routes_2_distance])

        routes_1_num_seed_routes = count_seed_routes(routes_1)
        routes_2_num_seed_routes = count_seed_routes(routes_2)

        routes_1_num_routes = len([route for route in routes_1 if route.num_vertices() > 1])
        routes_2_num_routes = len([route for route in routes_2 if route.num_vertices() > 1])
        

        results['num_seed_routes'].append([routes_1_num_seed_routes, routes_2_num_seed_routes])
        results['total_num_routes'].append([routes_1_num_routes, routes_2_num_routes])
        print(results)

        enablePrint()
        

        sys.stdout.write('\r')
        sys.stdout.write("[%-30s] %d%%" % ('='*10*(i+1), 100*(i+1)/N_SAMPLES))
        sys.stdout.flush()
        time.sleep(0.25)


    enablePrint()
    print('Done')
    print(results)
    # print a new line
    print()

    with open('results.json', 'w') as f:
        json.dump(results, f)


    if GRAPH_ROUTES:
        plot_routes(routes_1, "grasp1")
        plot_routes(routes_2, "grasp2") 
        plt.show()


def main5():
    order_quantities = [10, 20, 30, 40, 50, 60, 70, 80]
    #create a list of orders
    for n_orders in order_quantities:
        results[n_orders] = dict()
        results[n_orders]['runtimes'] = list()
        results[n_orders]['distances'] = list()

    
        for i in range(N_SAMPLES):
            blockPrint()

            orders_1 = create_orders(n_orders)

            start = time.time()
            routes_1 = grasp1(orders_1, False)
            routes_1_time = round(time.time() - start, 3)
            routes_1_distance = round(get_overall_distance(routes_1), 3)
           
            results[n_orders]['runtimes'].append(routes_1_time)
            results[n_orders]['distances'].append(routes_1_distance)

            print(results)

            enablePrint()
            

            sys.stdout.write('\r')
            sys.stdout.write("[%-30s] %d%%" % ('='*10*(i+1), 100*(i+1)/N_SAMPLES))
            sys.stdout.flush()
            time.sleep(0.25)


    enablePrint()
    print('Done')
    print(results)
    # print a new line
    print()

    with open('results.json', 'w') as f:
        json.dump(results, f)


    if GRAPH_ROUTES:
        plot_routes(routes_1, "grasp1")
        plot_routes(routes_2, "grasp2") 
        plt.show()


def create_orders(quantity):
    orders = list()
    for i in range (1, quantity+1):

        random_hour = random.uniform(1, 2.5)
        
        # orders scheduled between 10:00 -> 18:00(delivery starts at 09:00)
        scheduled_time = datetime.timedelta(hours=randrange(10, 16), minutes=randrange(0, 59))

        time_to_delivery = (scheduled_time-datetime.timedelta(hours=9))

        coords = (random.randrange(0,300),random.randrange(0,300))
        # scheduled_time = random.randrange(9, 6)
        orders.append(Order(i, 'ray', coords, scheduled_time, randrange(30)))

    return orders


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def count_seed_routes(routes):

    num_seed_routes = 0

    for route in routes:
        if len([vertex for vertex in route.vertices() if vertex.seed]) == 1:
            num_seed_routes += 1

    return num_seed_routes


if __name__ == '__main__':

    #sys.stdout = open(os.devnull, 'w')
    '''
    while True:
        if not main():
            plt.show()
            break
    '''
    main()
    '''
    orders = create_orders(NUMBER_OF_ORDERS)
    routes = grasp_VRPTW(orders, False)
    while routes_are_feasible(routes):

        routes = grasp_VRPTW(orders, False)
        print('adjust speed')
        time.sleep(3)

        # with is like your try .. finally block in this case
        with open('config.txt', 'r') as file:
            # read a list of lines into data
            lines = file.readlines()

        lines[1] = lines[1].split()[0] + ' ' + str(int(lines[1].split()[1]) - 1) + '\n'

        with open('config.txt', 'w') as file:
            file.writelines(lines)



    plot_routes(routes, "DCVRP")

    print(routes_are_feasible(routes))
    print(get_overall_distance(routes))

    plt.show()
    '''