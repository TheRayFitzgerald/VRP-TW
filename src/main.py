from math import sqrt
from random import random, randrange
from operator import itemgetter, attrgetter
from VRPTW.grahamscan import GrahamScan
from DCVRP.GRASP import grasp as grasp_DCVRP
from VRPTW.GRASP import grasp as grasp_VRPTW, Order, plot_routes, routes_are_feasible, get_overall_distance
from read_config import read_config
import matplotlib.pyplot as plt
import random, pickle, datetime, time, os, sys
import numpy as np

all_couriers = list()
all_orders = list()
all_users = list()
all_vans = list()

config_vars = read_config()
for key,val in config_vars.items():
    exec(key + '=val')

def main():

    #create a list of orders
    orders = create_orders(NUMBER_OF_ORDERS)

    routes_DCVRP = grasp_DCVRP(orders, False)
    routes_VRPTW = grasp_VRPTW(orders, False)
    plot_routes(routes_DCVRP, "DCVRP 1")
    plot_routes(routes_DCVRP, "DCVRP 2")
    plot_routes(routes_VRPTW, "VRPTW 1")
    plot_routes(routes_VRPTW, "VRPTW 2")

    print(routes_are_feasible(routes_DCVRP))
    print(get_overall_distance(routes_DCVRP))
    print(routes_are_feasible(routes_VRPTW))
    print(get_overall_distance(routes_VRPTW))

    plt.show()


    return routes_are_feasible(routes_DCVRP)

    
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


if __name__ == '__main__':

    #sys.stdout = open(os.devnull, 'w')
    '''
    while True:
        if not main():
            plt.show()
            break
    '''
    #main()
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


