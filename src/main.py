from Van import Van
from Courier import Courier
from Order import Order
from random import random, randrange
from operator import itemgetter
from grahamscan import GrahamScan
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
    orders = create_orders(10)

    print('\nUnsorted orders: ')
    for order in orders:
        print(order)

    print('\nsorted orders: ')
    orders_gch = greedy_construction(orders)
    for order in orders_gch:
        print(order[0])


    L = GrahamScan(orders)
    print(L)


def create_orders(quantity):
    orders = list()
    for i in range (0, quantity):
        coords = (np.random.randint(0,300),np.random.randint(0,300))
        orders.append(Order(i, 'ray', coords, '13:05', '14:30', randrange(30)))

    return orders

def greedy_construction(orders):

    unsorted_list = list()

    # create sorted list
    for order in orders:
        unsorted_list.append([order, order.distance + order.time_to_delivery])

    return sorted(unsorted_list, key=itemgetter(1))

def calculate_time_ratingt(time_val):
    current_time = (9, 0)
    # best 3 hours, worst 0
    # return a value with 0 as best, 1 as worst
    # 180 minutes => 100 / 180
    if time_val > 0:

        difference_hours = (time_val.split(':')[0] - current_time[0]) * 60
        difference_mins = time_val.split(':')[1] - current_
    100/12


if __name__ == '__main__':

    main()