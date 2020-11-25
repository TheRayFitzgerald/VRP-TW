from Van import Van
from Courier import Courier
from Order import Order
from math import sqrt
from random import random, randrange
from operator import itemgetter, attrgetter
from grahamscan import GrahamScan
from GRASP import grasp
import matplotlib.pyplot as plt
import random
import numpy as np

all_couriers = list()
all_orders = list()
all_users = list()
all_vans = list()



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
    orders = create_orders(100)
    grasp(orders)
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
    for i in range (1, quantity):
        coords = (np.random.randint(0,300),np.random.randint(0,300))
        orders.append(Order(i, 'ray', coords, '13:05', '14:30', randrange(30)))

    return orders




if __name__ == '__main__':

    main()