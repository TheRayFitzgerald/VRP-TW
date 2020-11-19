from Van import Van
from Courier import Courier
from Order import Order
from math import sqrt
from random import random, randrange
from operator import itemgetter, attrgetter
from grahamscan import GrahamScan
import matplotlib.pyplot as plt

import numpy as np

all_couriers = list()
all_orders = list()
all_users = list()
all_vans = list()

DEPOT_COORDS = (100, 100)

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
    greedy_construction(orders)
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
    for i in range (0, quantity):
        coords = (np.random.randint(0,300),np.random.randint(0,300))
        orders.append(Order(i, 'ray', coords, '13:05', '14:30', randrange(30)))

    return orders

def sort_L_comp(L_comp):
    unsorted_list = list()

def get_furthest_vertex(orders):

    max = orders[0]

    for order in orders:
        if order.distance > max.distance:
            max = order

    return max

def get_distance_between_vertices(vertex1, vertex2):

    return sqrt((vertex1[0] - vertex2[0])**2 + (vertex1[1] - vertex2[1])**2)

def route_initialization(L, L_comp):

    # sort by shortest distance from depot
    L_comp.sort(key=lambda x: x.distance, reverse=True)

    S = list()
    S.append(max(L, key=attrgetter('distance')))

    while len(S) < 6:

        # find order that maximises sum of distances from existing seeds in S
        j = (None, 0)
        for order in L:
            accum_distance = 0
            for seed in S:
                accum_distance += get_distance_between_vertices(seed.coords, order.coords)

            if accum_distance > j[1]:
                j = (order, accum_distance)
        j = j[0]
        print(j.distance)

        # get min distance between j and seeds
        min_distance_to_j = None
        for seed in S:
            if min_distance_to_j == None:
                min_distance_to_j = get_distance_between_vertices(seed.coords, j.coords)
            if get_distance_between_vertices(seed.coords, j.coords) < min_distance_to_j:
                min_distance_to_j = get_distance_between_vertices(seed.coords, j.coords)

        # get i, first item of L_comp

        i = L_comp[0]

        # get min distance between i and seeds
        min_distance_to_i = None
        for seed in S:
            if min_distance_to_i == None:
                min_distance_to_i = get_distance_between_vertices(seed.coords, i.coords)
            if get_distance_between_vertices(seed.coords, i.coords) < min_distance_to_i:
                min_distance_to_i = get_distance_between_vertices(seed.coords, i.coords)

        print('min %f' % min_distance_to_i)


        if min_distance_to_i < min_distance_to_j:
            L.remove(j)
            S.append(j)
        elif min_distance_to_i > min_distance_to_j:
            L_comp.remove(i)
            S.append(i)
    return S


def greedy_construction(orders):

    L_points, L = GrahamScan(orders)
    L_comp = list(set(orders) - set(L))
    # plot_convex_hull(orders, L_points)
    S = route_initialization(L, L_comp)

    plot_convex_hull(S, L_points)




def plot_convex_hull(orders, L):

    P = list()
    for order in orders:
        P.append(order.coords)
    P = np.array(P)


    plt.figure()
    #plt.plot(cplr)
    #plt.plot(DEPOT_COORDS[:,0],DEPOT_COORDS[:,1], 'b-', picker=5)
    plt.plot(DEPOT_COORDS[0], DEPOT_COORDS[1], 'ro')
    plt.plot(L[:,0],L[:,1], 'b-', picker=5)
    plt.plot([L[-1,0],L[0,0]],[L[-1,1],L[0,1]], 'b-', picker=5)
    plt.plot(P[:,0],P[:,1],".r")
    plt.axis('off')
    plt.show()


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