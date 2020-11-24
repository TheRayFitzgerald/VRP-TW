from Van import Van
from Courier import Courier
from Order import Order
from math import sqrt
from random import random, randrange
from operator import itemgetter, attrgetter
from grahamscan import GrahamScan
from Graph import Graph, Vertex
import matplotlib.pyplot as plt
import random, operator
import numpy as np


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

    # check this line with KB
    L.remove(max(L, key=attrgetter('distance')))

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

        i = (None, 0)
        for order in L_comp:
            accum_distance = 0
            for seed in S:
                accum_distance += get_distance_between_vertices(seed.coords, order.coords)
            if accum_distance > i[1]:
                i = (order, accum_distance)
        i = i[0]

        # get min distance between i and seeds
        min_distance_to_i = None
        for seed in S:
            if min_distance_to_i == None:
                min_distance_to_i = get_distance_between_vertices(seed.coords, i.coords)
            if get_distance_between_vertices(seed.coords, i.coords) < min_distance_to_i:
                min_distance_to_i = get_distance_between_vertices(seed.coords, i.coords)

        print('min %f' % min_distance_to_i)


        if min_distance_to_i < min_distance_to_j:
            print('j')
            S.append(L.pop(L.index(j)))
        elif min_distance_to_i > min_distance_to_j:
            print('i')
            S.append(L_comp.pop(L_comp.index(i)))

    routes = Graph()
    for order in S:
        routes.add_vertex(order)

    return [S, L, L_comp, routes]

def calculate_penalty(order, min_cost, other_routes):

    penalty = 0
    for route in other_routes:
        penalty += (get_distance_between_vertices(order.coords, route.coords) - min_cost)

    return penalty

def main_routing(unscheduled_orders, S, routes):

    # find the min insertion cost and the best route
    while(len(unscheduled_orders) > 0):
        order_rcps = list()
        for order in unscheduled_orders:
            route_cost_penalty = None

            for route in S:
                if route_cost_penalty == None:
                    route_cost_penalty = [route, get_distance_between_vertices(route.coords, order.coords), None]
                if get_distance_between_vertices(route.coords, order.coords) < route_cost_penalty[1]:
                    route_cost_penalty = [route, get_distance_between_vertices(route.coords, order.coords), None]

            other_routes = S.copy()
            other_routes.remove(route_cost_penalty[0])
            # best route found. now caculate the penalty
            route_cost_penalty[2] = calculate_penalty(order, route_cost_penalty[1], other_routes)
            order_rcps.append([order, route_cost_penalty])
            #print(route_cost_penalty[0].id)

            #construct fractional sublist of

        if(len(order_rcps) >= 5):
            largest_penalty_orders = sorted(order_rcps, key=lambda x: x[1][2], reverse=True)[:(len(order_rcps) // 5)]
        else:
            largest_penalty_orders = sorted(order_rcps, key=lambda x: x[1][2], reverse=True)

        added_order_rcp = random.choice(largest_penalty_orders)
        added_order = routes.add_vertex(added_order_rcp[0])
        routes.add_edge(added_order,added_order_rcp[1][0], added_order_rcp[1][1])

        unscheduled_orders.remove(added_order_rcp[0])

    return routes

def grasp(orders):

    L_points, L = GrahamScan(orders)
    L_comp = list(set(orders) - set(L))
    #plot_convex_hull(orders, L_points, '.r')
    S, L, L_comp, routes = route_initialization(L, L_comp)
    unscheduled_orders = L + L_comp

    #plot_convex_hull(S, L_points, 'bo')

    routes = main_routing(unscheduled_orders, S, routes)


def plot_convex_hull(orders, L, colour):

    P = list()
    for order in orders:
        P.append(order.coords)
    P = np.array(P)


    plt.figure()
    #plt.plot(cplr)
    #plt.plot(DEPOT_COORDS[:,0],DEPOT_COORDS[:,1], 'b-', picker=5)
    plt.plot(DEPOT_COORDS[0], DEPOT_COORDS[1], '*g')
    plt.plot(L[:,0],L[:,1], 'b-', picker=5)
    plt.plot([L[-1,0],L[0,0]],[L[-1,1],L[0,1]], 'b-', picker=5)
    plt.plot(P[:,0],P[:,1], colour)
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

