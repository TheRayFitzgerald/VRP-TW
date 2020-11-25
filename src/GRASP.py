from Van import Van
from Courier import Courier
from Order import Order
from math import sqrt
from random import random, randrange
from operator import itemgetter, attrgetter
from grahamscan import GrahamScan
from Graph import Graph, Vertex
import matplotlib.pyplot as plt
import random, operator, time
import numpy as np
from collections import Counter

DEPOT_COORDS = (150, 150)

def sort_L_comp(L_comp):
    unsorted_list = list()

def get_furthest_vertex(orders):

    max = orders[0]

    for order in orders:
        if order.distance > max.distance:
            max = order

    return max

def get_nearest_vertex(origin_vertex, vertices):

    min = vertices[0]

    for vertex in vertices:
        try:
            if get_distance_between_vertices(origin_vertex.element().coords, vertex.element().coords) < get_distance_between_vertices(origin_vertex.element().coords, min.element().coords):
                min = vertex
        except:
            pass


    return min

def get_distance_between_vertices(vertex1, vertex2):

    return sqrt((vertex1[0] - vertex2[0])**2 + (vertex1[1] - vertex2[1])**2)

def route_initialization(L, L_comp):

    # sort by shortest distance from depot
    L_comp.sort(key=lambda x: x.distance, reverse=True)

    S = list()
    S.append(max(L, key=attrgetter('distance')))

    # check this line with KB
    L.remove(max(L, key=attrgetter('distance')))

    while len(S) < 3:

        # find order that maximises sum of distances from existing seeds in S
        j = (None, 0)
        for order in L:
            accum_distance = 0
            for seed in S:
                accum_distance += get_distance_between_vertices(seed.coords, order.coords)
            if accum_distance > j[1]:
                j = (order, accum_distance)
        j = j[0]

        # get min distance between j and seeds
        min_distance_to_j = None
        for seed in S:
            if min_distance_to_j == None:
                min_distance_to_j = get_distance_between_vertices(seed.coords, j.coords)
            if get_distance_between_vertices(seed.coords, j.coords) < min_distance_to_j:
                min_distance_to_j = get_distance_between_vertices(seed.coords, j.coords)

        # get i, first item of L_comp

        i = None

        for order in L_comp:
            accum_distance = 0
            for seed in S:
                accum_distance += get_distance_between_vertices(seed.coords, order.coords)
            if i == None:
                i = (order, accum_distance)
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

        if min_distance_to_i < min_distance_to_j:
            S.append(L.pop(L.index(j)))
        elif min_distance_to_i > min_distance_to_j:
            S.append(L_comp.pop(L_comp.index(i)))

    routes = dict()
    for seed in S:
        routes[seed] = Graph()
        routes[seed].add_vertex(Order(0, 'ray', DEPOT_COORDS, '00:00', '00:00', 0))
        routes[seed].add_vertex(seed)


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

            #construct fractional sublist of

        if(len(order_rcps) >= 5):
            largest_penalty_orders = sorted(order_rcps, key=lambda x: x[1][2], reverse=True)[:(len(order_rcps) // 5)]
        else:
            largest_penalty_orders = sorted(order_rcps, key=lambda x: x[1][2], reverse=True)

        added_order_rcp = random.choice(largest_penalty_orders)
        added_order = routes[added_order_rcp[1][0]].add_vertex(added_order_rcp[0])
        #routes[added_order_rcp[1][0]].add_edge(added_order, routes[added_order_rcp[1][0]].get_vertex(added_order_rcp[1][0]), int(round(added_order_rcp[1][1], 0)))


        unscheduled_orders.remove(added_order_rcp[0])

    # routes now filled with vertices
    # construct the best edges

    for route in routes.values():

        remaining_vertices = route.vertices()[1:].copy()
        depot = route.vertices()[0]
        nearest_vertex = get_nearest_vertex(depot, remaining_vertices)
        route.add_edge(depot, nearest_vertex, get_distance_between_vertices(depot.element().coords, nearest_vertex.element().coords))

        while(len(remaining_vertices) > 1):

            for vertex in remaining_vertices:

                if len(route.get_edges(vertex)) == 1:
                    remaining_vertices.remove(vertex)
                    nearest_vertex = get_nearest_vertex(vertex, remaining_vertices)
                    edge = route.add_edge(vertex, nearest_vertex, get_distance_between_vertices(vertex.element().coords, nearest_vertex.element().coords))
                    break

            if len(remaining_vertices) == 0:
                break

        route.add_edge(remaining_vertices[0], route.vertices()[0], get_distance_between_vertices(remaining_vertices[0].element().coords, route.vertices()[0].element().coords))

    return routes

def grasp(orders):

    L_points, L = GrahamScan(orders)
    L_comp = list(set(orders) - set(L))
    #plot_convex_hull(orders, L_points, '.r')
    S, L, L_comp, routes = route_initialization(L, L_comp)
    unscheduled_orders = L + L_comp

    #plot_convex_hull(S, L_points, 'bo')

    routes = main_routing(unscheduled_orders, S, routes)


    #print(routes)
    #plot_convex_hull(routes.vertices(), L_points, 'bo')
    #print(routes.values().num_edges())
    plt.figure()
    P = list()
    for order in orders:
        P.append(order.coords)
    plt.scatter(*zip(*P))
    P = np.array(P)



    # plt.plot(P[:,0],P[:,1], 'b-', picker=5)
    #plt.plot([P[-1,0],P[0,0]],[P[-1,1],P[0,1]], 'b-', picker=5)
    #plt.plot(P[:,0],P[:,1], 'b')
    plt.plot(DEPOT_COORDS[0], DEPOT_COORDS[1], '*g')
    for route in routes.values():
        for edge in route.edges():
            vertices = edge.vertices()
            try:
                x_values = [vertices[0].element().coords[0], vertices[1].element().coords[0]]
                y_values = [vertices[0].element().coords[1], vertices[1].element().coords[1]]
            except AttributeError:
                x_values = [vertices[0].element().coords[0], vertices[1].element()[0]]
                y_values = [vertices[0].element().coords[1], vertices[1].element()[1]]
            plt.plot(x_values, y_values)


        #print(vertices[0].element().id)
        #plt.plot(vertices[0].element().coords, vertices[1].element().coords)
    #plt.axis('off')
    plt.show()

def plot_convex_hull(orders, L, colour):

    P = list()
    for order in orders:
        P.append(order.element().coords)
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

