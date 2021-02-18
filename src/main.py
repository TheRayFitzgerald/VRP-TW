from Van import Van
from Courier import Courier
from Order import Order
from math import sqrt
from random import random, randrange
from operator import itemgetter, attrgetter
from grahamscan import GrahamScan
from GRASP import grasp, calculate_slack, order_is_reachable, routes_are_feasible, tw_shuffle
import matplotlib.pyplot as plt
import random, pickle, datetime, time
import numpy as np

all_couriers = list()
all_orders = list()
all_users = list()
all_vans = list()

NUMBER_OF_ORDERS = 40

def main():
    #create a new van and courier
    '''
    van1 = Van(100)
    courier1 = Courier('Mike')
    courier1.van = van1
    print(courier1)
    order1 = Order(12, 'ray', 'Main Road, Cork', '13:05', '14:30', randrange(30))
    print(order1)
    '''

    #create a list of orders
    orders = create_orders(NUMBER_OF_ORDERS)

    #with open('orders.pkl', 'wb') as output:
     #   pickle.dump(orders, output, pickle.HIGHEST_PROTOCOL)

    for order in orders:
        
        time_to_delivery = (order.scheduled_time-datetime.timedelta(hours=9))

        print('###')
        print(order.scheduled_time)
        # print(time_to_delivery.total_seconds())
        print(order.distance)
        print(order.slack)
        print(order.scheduled_time.seconds)
        print('###')
        
    print(calculate_slack(orders[-1]))
    routes = grasp(orders, False)
    
    '''

    is_reachable_count = 0
    not_reachable_count = 0

    for route in routes:
        for vertex in route.vertices():
            if order_is_reachable(route, vertex):
                is_reachable_count += 1
            else:
                print(vertex.order.slack)
                #time.sleep(2)
                not_reachable_count += 1

    print(is_reachable_count, not_reachable_count)
    print('ya')

    print('$$$')
    print(routes_are_feasible(routes))
    
    a = input("Would you like to save these Routes? [y/n]: ")
    if a == 'y':
        with open('routes_4.pkl', 'wb') as output:
            pickle.dump(routes, output, pickle.HIGHEST_PROTOCOL)
            print('Saved')
    else:
        print('Exiting')
    
    '''
        
    '''
    print('\nUnsorted orders: ')
    for order in orders:
        print(order)

    print('\nsorted orders: ')
    orders_gch = greedy_construction(orders)
    for order in orders_gch:
        print(order[0])
    '''



def create_orders(quantity):
    orders = list()
    for i in range (quantity):

        random_hour = random.uniform(1, 2.5)
        
        # orders scheduled between 10:00 -> 18:00(delivery starts at 09:00)
        scheduled_time = datetime.timedelta(hours=randrange(10, 18), minutes=randrange(0, 59))

        time_to_delivery = (scheduled_time-datetime.timedelta(hours=9))

        coords = (random.randrange(0,300),random.randrange(0,300))
        # scheduled_time = random.randrange(9, 6)
        orders.append(Order(i, 'ray', coords, scheduled_time, randrange(30)))


        

    return orders


if __name__ == '__main__':

    for i in range(1):
        main()