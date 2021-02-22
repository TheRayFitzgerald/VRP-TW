from Van import Van
from Courier import Courier
from Order import Order
from math import sqrt
from random import random, randrange
from operator import itemgetter, attrgetter
from grahamscan import GrahamScan
from Graph import Graph, Vertex
import matplotlib.pyplot as plt
import random, operator, time, copy, pickle, datetime
import numpy as np
from collections import Counter

from py2opt.routefinder import RouteFinder

DEPOT_COORDS = (150, 150)
START_TIME = datetime.timedelta(hours=9)
MAX_NUMBER_OF_ROUTES = 6
SPEED = 20

local_search_actioned = False
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

def sort_L_comp(L_comp):
    unsorted_list = list()

def calculate_slack(order):
    return order.scheduled_time - (START_TIME + datetime.timedelta(minutes=round(order.distance / SPEED))) 

def is_completed_route(route):

    if route.vertices()[0] == route.vertices()[-1]:
        print('Complete Route')

    else:
        print('Incomplete Route')
        print(route.vertices()[0])
        print(route.vertices()[-1])

def is_completed_route2(route):

    if route.edges()[0].start() == route.edges()[-1].end():
        print('Complete Route')        
    else:
        print('Incomplete Route')

def order_is_reachable(route, order):
    total_time = 0
    #print('-- Order: %i' % order.order.id)
    for edge in route.edges_in_order():
        #print(edge)
        start = edge.start()
        end = edge.end()

        total_time += route.get_distance_between_vertices(start, end) / SPEED

        if end == order:
            break

    total_time = datetime.timedelta(minutes=total_time)

    return START_TIME + total_time < order.element().scheduled_time

def route_is_feasible(route):

    for vertex in route.vertices()[1:]:
        if not order_is_reachable(route, vertex):
            return False

    return True

def routes_are_feasible(routes):

    for route in routes:
        if not route_is_feasible(route):
            return False

    return True


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

def get_route_distance(route):

    total_distance = 0

    for edge in route.edges():
        total_distance += edge.element()

    return total_distance

def get_overall_distance(routes):

    overall_distance = 0

    for route in routes:
        overall_distance += get_route_distance(route)

    return overall_distance


def route_initialization_0(L, L_comp):

    # sort by shortest distance from depot
    L_comp.sort(key=lambda x: x.slack, reverse=True)

    S = list()
    S.append(max(L, key=attrgetter('slack')))

    # check this line with KB
    L.remove(max(L, key=attrgetter('slack')))

    while len(S) < MAX_NUMBER_OF_ROUTES:

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

    routes = list()
    i = 0
    for seed in S:
        g = Graph(i)
        g.add_vertex(Order(0, 'ray', DEPOT_COORDS, START_TIME, 0))
        g.add_vertex(seed, seed=True)
        routes.append(g)
        #routes[len(routes) - 1].add_vertex(Order(0, 'ray', DEPOT_COORDS, '00:00', '00:00', 0))
        i += 1


    return [S, L, L_comp, routes]

def route_initialization(L, L_comp):

    # sort by shortest distance from depot
    L_comp.sort(key=lambda x: x.slack, reverse=True)

    S = list()
    S.append(max(L, key=attrgetter('slack')))

    # check this line with KB
    L.remove(max(L, key=attrgetter('slack')))

    while len(S) < MAX_NUMBER_OF_ROUTES:

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

    routes = list()
    i = 0
    for seed in S:
        g = Graph(i)
        depot = g.add_vertex(Order(0, 'ray', DEPOT_COORDS, START_TIME, 0))
        seed = g.add_vertex(seed, seed=True)
        g.add_edge(depot, seed)
        routes.append(g)
        #routes[len(routes) - 1].add_vertex(Order(0, 'ray', DEPOT_COORDS, '00:00', '00:00', 0))
        i += 1


    return [S, L, L_comp, routes]

def calculate_penalty(order, min_cost, other_routes):

    penalty = 0
    for route in other_routes:
        penalty += (get_distance_between_vertices(order.coords, route.coords) - min_cost)

    return penalty

def get_distance_matrix(route):

    distance_matrix = list()
    for origin_vertex in route.vertices():
        distances_for_vertex = list()

        for destination_vertex in route.vertices():    
            distance = get_distance_between_vertices(origin_vertex.element().coords, destination_vertex.element().coords)
            distances_for_vertex.append(distance)

        distance_matrix.append(distances_for_vertex)

    return distance_matrix

def iterate_routes(routes, function):

    for route in routes:
        for order in routes:
            pass

def main_routing_0(unscheduled_orders, S, routes):

    # find the min insertion cost and the best route
    print('\n######\nMain Routing\n######\n')
    
    
    while(len(unscheduled_orders) > 0):
        #print(len(unscheduled_orders))
        order_rcps = list()
        for order in unscheduled_orders:
            route_cost_penalty = None
            min_cost_for_routes = list()

            for route in routes:

                min_cost_for_route = None

                for vertex in route.vertices():
                    if route.degree(vertex) < 2:

                        # get the optimum route and its associated cost.
                        if route_cost_penalty == None:
                            route_cost_penalty = [route, get_distance_between_vertices(vertex.element().coords, order.coords), None, vertex]
                        if get_distance_between_vertices(vertex.element().coords, order.coords) < route_cost_penalty[1]:
                            route_cost_penalty = [route, get_distance_between_vertices(vertex.element().coords, order.coords), None, vertex]

                        # get minimum cost for each route. To be used in calculating penalty.
                        if min_cost_for_route == None:
                            min_cost_for_route = [route, get_distance_between_vertices(vertex.element().coords, order.coords)]
                        elif get_distance_between_vertices(vertex.element().coords, order.coords) < min_cost_for_route[1]:
                            min_cost_for_route = [route, get_distance_between_vertices(vertex.element().coords, order.coords)]
                min_cost_for_routes.append(min_cost_for_route)

            # best route for given order found.
            # now caculate the penalty using the best cost for other routes
            other_routes = routes.copy()
            other_routes.remove(route_cost_penalty[0])
            route_cost_penalty[2] = 0
            for min_cost_for_route in min_cost_for_routes:
                if min_cost_for_route[0] != route_cost_penalty[0]:
                    route_cost_penalty[2] += min_cost_for_route[1]

            # add to list of all order,root cost penalties
            order_rcps.append([order, route_cost_penalty])

        # get largest penalty orders
        largest_penalty_orders = sorted(order_rcps, key=lambda x: x[1][2], reverse=True)
        added_order_rcp = largest_penalty_orders[0]
        
        # now route this order
        added_order = routes[routes.index(added_order_rcp[1][0])].add_vertex(added_order_rcp[0])
        print('added order')
        print(added_order)

        time.sleep(3)
        # add edge between existing vertex
        try:
            added_edge = routes[routes.index(added_order_rcp[1][0])].add_edge(added_order_rcp[1][3], added_order)
            print(added_edge)
            time.sleep(3)
            #routes[routes.index(added_order_rcp[1][0])].add_edge(routes[routes.index(added_order_rcp[1][0])].vertices()[len(routes[routes.index(added_order_rcp[1][0])].vertices()) - 3], added_order, 3)
        except Exception as e:
            print(e)
            time.sleep(3)
        

        unscheduled_orders.remove(added_order_rcp[0])

    # routes now filled with vertices
    # construct the best edges
    '''
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
    '''

    return routes

def main_routing(unscheduled_orders, S, routes):

    # find the min insertion cost and the best route
    print('\n######\nMain Routing\n######\n')
    for route in routes:

        print('Route')
        print(route)
        for edge in route.edges():
            print('EDGES')
            print(edge)
            #time.sleep(3)

    while(len(unscheduled_orders) > 0):
        #print(len(unscheduled_orders))
        order_rcps = list()
        for order in unscheduled_orders:
            route_cost_penalty = None
            min_cost_for_routes = list()

            for route in routes:
                min_cost_for_route = None

                for edge in route.edges():
                    
                    start = edge.start()
                    end = edge.end()
                    order_v = route.add_vertex(order)
                    route.add_vertex_between_vertices(order_v, start, end)
                    if route_is_feasible(route):

                        d3 = route.get_distance_between_vertices(start, end)
                        d1 = get_distance_between_vertices(start.order.coords, order.coords)
                        d2 = get_distance_between_vertices(order.coords, end.order.coords)

                        cost = d1 + d2 - d3
                        
                        # get the optimum route and its associated cost.
                        if route_cost_penalty == None or cost < route_cost_penalty[1]:
                            route_cost_penalty = [route, cost, None, edge]

                        # get minimum cost for each route. To be used in calculating penalty.
                        if min_cost_for_route == None:
                            min_cost_for_route = [route, cost]
                        elif cost < min_cost_for_route[1]:
                            min_cost_for_route = [route, cost]

                    route.remove_vertex_and_repair(order_v)
            min_cost_for_routes.append(min_cost_for_route)

            # best route for given order found.
            # now caculate the penalty using the best cost for other routes
            other_routes = routes.copy()
            other_routes.remove(route_cost_penalty[0])
            route_cost_penalty[2] = 0
            for min_cost_for_route in min_cost_for_routes:
                if min_cost_for_route[0] != route_cost_penalty[0]:
                    route_cost_penalty[2] += min_cost_for_route[1]

            # add to list of all order,root cost penalties
            order_rcps.append([order, route_cost_penalty])

        # get largest penalty orders
        largest_penalty_orders = sorted(order_rcps, key=lambda x: x[1][2], reverse=True)
        added_order_rcp = largest_penalty_orders[0]
        
        # now route this order
        added_order = routes[routes.index(added_order_rcp[1][0])].add_vertex(added_order_rcp[0])

        
        # add edge between existing vertex
        try:
            #added = routes[routes.index(added_order_rcp[1][0])].add_edge(added_order_rcp[1][3], added_order)
            w1 = added_order_rcp[1][3].start()
            w2 = added_order_rcp[1][3].end()
            added = routes[routes.index(added_order_rcp[1][0])].add_vertex_between_vertices(added_order, w1, w2)

            print('TSE')
            print(added)
            #time.sleep(1)
            #routes[routes.index(added_order_rcp[1][0])].add_edge(routes[routes.index(added_order_rcp[1][0])].vertices()[len(routes[routes.index(added_order_rcp[1][0])].vertices()) - 3], added_order, 3)
        except Exception as e:
            print(e)
        

        unscheduled_orders.remove(added_order_rcp[0])

    for route in routes:
        route.complete_route()
    return routes

# takes routes and utilises 2-opt to imrpove the routes
def two_opt_route_improve(routes):

    improved_routes = list()
    for i in range(len(routes)):

        dist_mat = get_distance_matrix(routes[i])
        route_finder = RouteFinder(dist_mat, routes[i].vertices(), iterations=10)
        best_distance, best_route = route_finder.solve()

        improved_route = Graph('4')

        for start, end in zip(best_route, best_route[1:]):

            v1 = improved_route.add_existing_vertex(routes[i].get_vertex(start.element()))
            v2 = improved_route.add_existing_vertex(routes[i].get_vertex(end.element()))

            improved_route.add_edge(v1, v2)
    
        improved_route.complete_route()
        improved_routes.append(improved_route)    
    
    return set_id_by_position(improved_routes)


def cost_of_break(order, route):

    original_distance = get_route_distance(route)
    route_copy = copy.deepcopy(route)

    original_edges = route.get_edges(order)
    print('Original Edges')
    print(original_edges)
    print('Order')
    print(order)

    if len(original_edges) > 1:

        v1 = original_edges[0].opposite(order)
        v2 = original_edges[1].opposite(order)

        
        d1 = route.get_distance_between_vertices(v1, order)
        d2 = route.get_distance_between_vertices(order, v2)

        d3 = route.get_distance_between_vertices(v1, v2)
    else:
        v1 = original_edges[0].opposite(order)
        d1 = route.get_distance_between_vertices(v1, order)
        d2 = d1
        # depot to depot
        d3 = 0

  
    return d3 - (d1 + d2)

def route_distance_difference(route1, route2):

    return get_route_distance(route1) - get_route_distance(route2)

def local_search(routes):

    local_search_actioned = False

    for i in range(len(routes)): 
        origin_route = routes[i]

        for order in origin_route.vertices()[1:]:

            other_routes = routes.copy()
            other_routes.pop(routes.index(origin_route))

            best_new_position_route = None
            # magic
            for j in range(len(other_routes)):
                improved_other_route = copy.deepcopy(other_routes[j])
                
                for edge in improved_other_route.edges():

                    start = edge.start()
                    end = edge.end()
  
                    print('$$$$ START $$$$\n')
                    print(improved_other_route)
                    print('Distance: %f' % get_route_distance(improved_other_route))
                
                    print('\n----- Origin -----')
                    print('Route: %s, Order: %i' % (colors[i], order.element().id))
                    print('------------------')

                    print('\n----- Destination -----')
                    print('Route: %s, Order: %i <-> %i' % (colors[routes.index(other_routes[j]) % 7], start.element().id, end.element().id))
                    print('-----------------------\n')

                    improved_routes = routes.copy()
                    improved_origin_route = copy.deepcopy(improved_routes[i])

                    if origin_route.degree(order) > 2:
                        # time.sleep(1)
                        print('TOO MANY EDGES')

                    print('$$$$ Origin Route $$$$\n')
                    print(origin_route)

                    print(improved_other_route.add_vertex_between_vertices(order, start, end))

                    print('\n$$$$ CHANGES $$$$')
                    print(improved_other_route)
                    print('Distance: %f' % get_route_distance(improved_other_route))
                    print('\n')

                    try:
                        # print('Best Route Distance: %f' % get_route_distance(best_new_position_route[0]))
                        print('Best Route Distance: %f' % best_new_position_route[4])
                        print('This Route Distance: %f' % get_route_distance(improved_other_route))
                    except Exception as e: 
                        print(e)


                    # find the new position with the lowest cost
                    if best_new_position_route == None or \
                    (get_route_distance(improved_other_route) < get_route_distance(best_new_position_route[0]) and \
                    route_distance_difference(improved_other_route, other_routes[j]) < route_distance_difference(best_new_position_route[0], other_routes[j])):
                        print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
                        print('\n2\n')
                        print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
                        
                        best_new_position_route = [copy.deepcopy(improved_other_route), j, start.element().uid, end.element().uid]
                    
                    # make reparations 
                    improved_other_route.remove_vertex_and_repair(order)

                    print(improved_other_route)
                    print('Distance: %f' % get_route_distance(improved_other_route))
                    print('$$$$ END $$$$\n')
    
            # check if dest route distance + cost of break in origin route is better than old dest route distance
            if (cost_of_break(order, origin_route) + get_route_distance(best_new_position_route[0])) \
                < (get_route_distance(routes[routes.index(other_routes[best_new_position_route[1]])])):

                local_search_actioned = True
                try:
                    routes[i].remove_vertex_and_repair(order)

                    #-------------------------------
                    r_index = routes.index(other_routes[best_new_position_route[1]])
                    
                    start = routes[r_index].get_vertex_by_uid(best_new_position_route[2])
                    end = routes[r_index].get_vertex_by_uid(best_new_position_route[3])
                    
                    # Add order to destination route
                    routes[r_index].add_vertex_between_vertices(order, start, end)

                except Exception as e:
                    print(e)
                    print('RAY')
                    print('ERROR')
                    print(';h')
                    time.sleep(5)
                    #plot_routes(routes, "error")
                    #plt.show()
                  
            else:
                print('fail')
                print(get_route_distance(origin_route) + cost_of_break(order, origin_route) + \
                get_route_distance(best_new_position_route[0]))
                print((get_route_distance(origin_route) + \
                get_route_distance(routes[routes.index(other_routes[best_new_position_route[1]])])))
                print(cost_of_break(order, origin_route))

    return (routes, local_search_actioned)

def two_opt_ls_iterator(routes, n_rounds):

    #with open("distances.txt", "a") as f:
     #   f.write("%f\n" % get_overall_distance(routes))
    timeout = time.time() + 90
    best_distance = get_overall_distance(routes)
    for i in range(n_rounds):
        while get_overall_distance(routes) >= best_distance:
            routes, local_search_actioned = local_search(routes)
            routes = two_opt_route_improve(routes)

            if time.time() > timeout:
                break

            #with open("distances.txt", "a") as f:
             #   f.write("%f\n" % get_overall_distance(routes))

        best_distance = get_overall_distance(routes)

    return routes

def tw_shuffle_local_search_iterator(routes, n_rounds):

    #with open("distances.txt", "a") as f:
     #   f.write("%f\n" % get_overall_distance(routes))
    timeout = time.time() + 90
    best_distance = get_overall_distance(routes)
    for i in range(n_rounds):
        while get_overall_distance(routes) >= best_distance:
            routes, failed_vertices = tw_shuffle(routes)
            routes = local_search_tw(routes)[0]
            

            if time.time() > timeout:
                return routes
                break

            #with open("distances.txt", "a") as f:
             #   f.write("%f\n" % get_overall_distance(routes))

        best_distance = get_overall_distance(routes)
        plot_routes(routes, "Round %i" % i)

    return routes

def local_search_tw_shuffle_iterator(routes, n_rounds):

    open('num_vertices.txt', 'w').close()
    num_vertices = 0
    for route in routes:
        num_vertices += route.num_vertices() - 1
    with open("num_vertices.txt", "a") as f:
        f.write("Initial: %i\n" % num_vertices)

    timeout = time.time() + 200
    best_distance = get_overall_distance(routes)
    for i in range(n_rounds):
        while get_overall_distance(routes) >= best_distance:
            routes = local_search_tw(routes)[0]
            routes, failed_vertices = tw_shuffle(routes)

            
        num_vertices = 0
        for route in routes:
            num_vertices += route.num_vertices() - 1
        with open("num_vertices.txt", "a") as f:
            f.write("%i\n" % num_vertices)

        best_distance = get_overall_distance(routes)
        plot_routes(routes, "Round %i" % (i+1))

    return routes

def local_search_tw_shuffle_iterator_2(routes):

    open('num_vertices.txt', 'w').close()
    num_vertices = 0
    for route in routes:
        num_vertices += route.num_vertices() - 1
    with open("num_vertices.txt", "a") as f:
        f.write("Initial: %i\n" % num_vertices)

    timeout = time.time() + 250
    best_distance = get_overall_distance(routes)

    while True:
        routes = local_search_tw(routes)[0]
        routes, failed_vertices = tw_shuffle(routes)

        
        num_vertices = 0
        for route in routes:
            num_vertices += route.num_vertices() - 1
        with open("num_vertices.txt", "a") as f:
            f.write("%i\n" % num_vertices)

        if time.time() > timeout:
            break

    best_distance = get_overall_distance(routes)
    plot_routes(routes, "Round x")

    return routes

def tw_shuffle(routes):

    failed_vertices = list()
    improved_routes = list()
    for route in routes:
        if route.num_vertices() > 2:
            
            failed_vertices_for_route = list()

            # create a new improved route with abritrary ID
            # this improved route will be used to record changes for all vertices for this given route.
            improved_route = Graph('4')

            shuffled_vertices = route.vertices()[1:]
            random.shuffle(shuffled_vertices)

            # add the depot to the start of the improved route
            d0 = improved_route.add_existing_vertex(route.vertices()[0])
            # add the first vertex from the shuffled vertices to begin
            d1 = improved_route.add_existing_vertex(shuffled_vertices.pop(0))
            # connect the first vertex to the depot
            improved_route.add_edge(d0, d1)
            
            # randomly pick a vertex to insert
            for vertex in shuffled_vertices:
                # create a list to record all feasible routes for this vertex.
                feasible_routes = list()
                # iterate through all edges to find the best position for this vertex
                for edge in improved_route.edges():
                    
                    start = edge.start()
                    end = edge.end()
                    improved_route.add_vertex_between_vertices(vertex, start, end)
                    
                    # check if route is feasible: all orders in route are reachable within their TW's.
                    if route_is_feasible(improved_route):
                        # add route to list of all feasible routes for this vertex
                        feasible_routes.append(copy.deepcopy(improved_route))
                    # undo changes
                    improved_route.remove_vertex_and_repair(vertex)
                    
                # all feasible routes have now been recorded
                # iterate through all feasible routes to find the cheapest
                try:
                    best_route = feasible_routes[0]
                    for route in feasible_routes[1:]:
                        if get_route_distance(route) < get_route_distance(best_route):
                            best_route = route

                    # make changes on the main improved route
                    improved_route = best_route

                # vertex inserted into its best position - continue

                # no routes found for this vertex
                except Exception as e:
                    print(e)
                    failed_vertices_for_route.append(vertex)
                    pass

            # complete the route
            improved_route.complete_route()
            # record the finalised improved route
            improved_routes.append(improved_route)
            failed_vertices.append(failed_vertices_for_route)
    for vertex in failed_vertices:
        print('###')
        print(vertex)
        print('###')
    return improved_routes, failed_vertices

def local_search_tw(routes):

    local_search_actioned = False

    shuffled_routes = routes
    random.shuffle(shuffled_routes)


    for i, origin_route in enumerate(shuffled_routes): 
        #origin_route = routes[i]

        shuffled_vertices = origin_route.vertices()[1:]
        random.shuffle(shuffled_vertices)

        for order in shuffled_vertices:

            other_routes = shuffled_routes.copy()
            other_routes.pop(shuffled_routes.index(origin_route))

            best_new_position_route = None
            # magic
            for j in range(len(other_routes)):
                improved_other_route = copy.deepcopy(other_routes[j])
                
                for edge in improved_other_route.edges():

                    start = edge.start()
                    end = edge.end()
  
                    print('$$$$ START $$$$\n')
                    print(improved_other_route)
                    print('Distance: %f' % get_route_distance(improved_other_route))
                
                    print('\n----- Origin -----')
                    print('Route: %s, Order: %i' % (colors[i % 7], order.element().id))
                    print('------------------')

                    print('\n----- Destination -----')
                    print('Route: %s, Order: %i <-> %i' % (colors[routes.index(other_routes[j]) % 7], start.element().id, end.element().id))
                    print('-----------------------\n')

                    improved_routes = routes.copy()
                    improved_origin_route = copy.deepcopy(improved_routes[i])

                    if origin_route.degree(order) > 2:
                        # time.sleep(1)
                        print('TOO MANY EDGES')

                    print('$$$$ Origin Route $$$$\n')
                    print(origin_route)

                    print(improved_other_route.add_vertex_between_vertices(order, start, end))

                    print('\n$$$$ CHANGES $$$$')
                    print(improved_other_route)
                    print('Distance: %f' % get_route_distance(improved_other_route))
                    print('\n')

                    try:
                        # print('Best Route Distance: %f' % get_route_distance(best_new_position_route[0]))
                        print('Best Route Distance: %f' % best_new_position_route[4])
                        print('This Route Distance: %f' % get_route_distance(improved_other_route))
                    except Exception as e: 
                        print(e)


                    # find the new position with the lowest cost
                    try:
                        if (best_new_position_route == None or \
                        (get_route_distance(improved_other_route) < get_route_distance(best_new_position_route[0]) and \
                        route_distance_difference(improved_other_route, other_routes[j]) < route_distance_difference(best_new_position_route[0], other_routes[j]))) \
                        and route_is_feasible(improved_other_route):
                            print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
                            print('\n2\n')
                            print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
                            
                            best_new_position_route = [copy.deepcopy(improved_other_route), j, start.element().uid, end.element().uid]
                        
                        # make reparations 
                        improved_other_route.remove_vertex_and_repair(order)

                        print(improved_other_route)
                        print('Distance: %f' % get_route_distance(improved_other_route))
                        print('$$$$ END $$$$\n')
                    except Exception as e:
                        print(e)
                        plot_routes(routes, "error on feasibility")
                        plt.show()
            
            
            if best_new_position_route == None:
                break
            # check if dest route distance + cost of break in origin route is better than old dest route distance
            if best_new_position_route != None and ((cost_of_break(order, origin_route) + get_route_distance(best_new_position_route[0])) \
                < (get_route_distance(routes[routes.index(other_routes[best_new_position_route[1]])]))):

                local_search_actioned = True
                try:
                    routes[i].remove_vertex_and_repair(order)

                    #-------------------------------
                    r_index = routes.index(other_routes[best_new_position_route[1]])
                    
                    start = routes[r_index].get_vertex_by_uid(best_new_position_route[2])
                    end = routes[r_index].get_vertex_by_uid(best_new_position_route[3])
                    
                    # Add order to destination route
                    routes[r_index].add_vertex_between_vertices(order, start, end)
                    
                    
                except Exception as e:
                    print(e)
                    print('ERROR')
                    time.sleep(5)
                    plot_routes(routes, "error")
                    plt.show()
                  
            else:
                print('fail')
                print(get_route_distance(origin_route) + cost_of_break(order, origin_route) + \
                get_route_distance(best_new_position_route[0]))
                print((get_route_distance(origin_route) + \
                get_route_distance(routes[routes.index(other_routes[best_new_position_route[1]])])))
                print(cost_of_break(order, origin_route))

    return (routes, local_search_actioned)

def grasp(orders, graph_results=True):

    L_points, L = GrahamScan(orders)
    L_comp = list(set(orders) - set(L))
    #plot_convex_hull(orders, L_points, '.r')
    S, L, L_comp, routes = route_initialization(L, L_comp)
    unscheduled_orders = L + L_comp
    
    #plot_convex_hull(S, L_points, 'bo')
    #plot_convex_hull(routes.vertices(), L_points, 'bo')

    routes_copy = copy.deepcopy(routes)
    routes = main_routing(unscheduled_orders, S, routes_copy)
    #routes_2 = main_routing_0(unscheduled_orders, S, routes)
    num_vertices=0
    for route in routes:
        num_vertices += route.num_vertices() - 1
    print('NUM VERTICES')
    print(num_vertices)
    print('FEASIBLE')
    print(routes_are_feasible(routes))

    if graph_results:
        # plot_routes(improved_routes, "Two-opt improved routes")
        plot_routes(routes, "main routing 1")
        #plot_routes(routes_2, "main routing 2")
        #plt.show()


    #return routes

    #routes_it = tw_shuffle_local_search_iterator(improved_routes, 3)
    #routes = local_search_tw_shuffle_iterator(routes, 7)
    routes = local_search_tw_shuffle_iterator_2(routes)

    num_vertices=0
    for route in routes:
        num_vertices += route.num_vertices() - 1
    print('NUM VERTICES')
    print(num_vertices)
    print('FEASIBLE')
    print(routes_are_feasible(routes))

    if graph_results:
        plot_routes(routes, "Iterator")
        plt.show()
    return routes

    
    for i in range(1):
        tw_shuffle_routes, failed_vertices = tw_shuffle(routes)


    for failed_vertices_for_route in failed_vertices:
        print('################')
        print('Route: %s' % colors[failed_vertices.index(failed_vertices_for_route)])
        for vertex in failed_vertices_for_route:
            print(vertex)
        print('################')

        
    if graph_results:
        plot_routes(tw_shuffle_routes, "TW Shuffle")

    #return tw_shuffle_routes
    for i in range(3):
        local_search_routes_tw = local_search_tw(tw_shuffle_routes)[0]

    if graph_results:
        plot_routes(local_search_routes_tw, "Local Search w/ TW")
        plt.show()

    print('Routes are Feasible: %s' % routes_are_feasible(tw_shuffle_routes))
    print('Routes are Feasible: %s' % routes_are_feasible(local_search_routes_tw))
    return local_search_routes_tw
    for i in range(5):
        improved_routes = two_opt_route_improve(tw_shuffle_routes)

    plot_routes(improved_routes, "2")
    plt.show()


    L_points, L = GrahamScan(orders)
    L_comp = list(set(orders) - set(L))
    #plot_convex_hull(orders, L_points, '.r')
    S, L, L_comp, routes = route_initialization(L, L_comp)
    unscheduled_orders = L + L_comp
    
    #plot_convex_hull(S, L_points, 'bo')
    #plot_convex_hull(routes.vertices(), L_points, 'bo')


    routes = main_routing(unscheduled_orders, S, routes)

    with open("distances.txt", "a") as f:
        f.write("%f\n" % get_overall_distance(routes))

    improved_routes = two_opt_route_improve(routes)
    #return improved_routes
    if graph_results:
        plot_routes(improved_routes, "2")


    

    with open('routes.pkl', 'wb') as output:
        pickle.dump(improved_routes, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(orders, output, pickle.HIGHEST_PROTOCOL)
    # plot_routes(improved_routes, "2-opt improved routes ## check")

    

    #with open("distances.txt", "a") as f:
     #   f.write("%f\n" % get_overall_distance(improved_routes))
    local_search_routes = improved_routes
    for i in range(3):
        local_search_routes = local_search(local_search_routes)[0]

    #local_search_routes, local_search_actioned = local_search(improved_routes.copy())
    if graph_results:
        plot_routes(local_search_routes, "Local Search")
        plt.show()


    for i in range(3):
        local_search_routes = two_opt_route_improve(local_search_routes)
    return local_search_routes
    #local_search_routes, local_search_actioned = local_search(improved_routes.copy())
    if graph_results:
        plot_routes(local_search_routes, "Two opt improve Search")
        plt.show()
    

    final_routes = two_opt_ls_iterator(local_search_routes, 1)
    with open("distances.txt", "a") as f:
        f.write("%f\n" % get_overall_distance(final_routes))
    if graph_results:
        plot_routes(final_routes, "Multiple Local Searches")
        plt.show()

    return final_routes
    '''
    with open("distances.txt", "a") as f:
        f.write("%f\n" % get_overall_distance(local_search_routes))
    

    print('\n\n')
    print(get_overall_distance(local_search_routes))
    print('\n\n')
    '''
    '''
    for route in improved_routes:
        print('\n##### NEW ROUTE######')
        for order in route.vertices():
            if len(route.get_edges(order)) != 2:
                print('BAD %i' % route.degree(order))
            elif len(route.get_edges(order)) == 2:
                print('!!! GOOD !!!')
            else:
                print(route.degree(order))
    '''
    '''
    
    # local_search_routes = local_search(improved_routes.copy())
    print(local_search_routes == improved_routes)
    # plot_routes(orders, routes, "initial greedy routes")
    final_routes = two_opt_ls_iterator(local_search_routes, 3)
    plot_routes(orders, final_routes, "2nd round two-opt improve")

    if local_search_actioned and 7 == 6:
        plot_routes(orders, local_search_routes, "local-search improved routes")

        final_routes = two_opt_ls_iterator(local_search_routes, 5)
        plot_routes(orders, final_routes, "2nd round two-opt improve")

        print('\n\n')
        print(get_overall_distance(final_routes))
        print('\n\n')
        with open("distances.txt", "a") as f:
            f.write("%f" % get_overall_distance(final_routes))

    with open("distances.txt", "a") as f:
            f.write("\n\n######\n\n")

    '''
    
def set_id_by_position(routes):
    for route in routes:
        for pos, order in enumerate(route.vertices()):
            order.element().id = pos
    
    return routes
    

def plot_routes(routes, title="untitled", labeled=True):


    fig = plt.figure()
    plt.suptitle(title, fontsize=20)
    ax = fig.add_subplot(111)

    P = list()
    C = list()
    for route in routes:
        for order in route.vertices():
            P.append(order.element().coords)
            # time_to_delivery = datetime.timedelta(hours=9) - order.order.scheduled_time
            C.append(np.cos(order.order.slack.total_seconds()))
            if labeled:
                plt.annotate(order.element().id, (order.element().coords[0], order.element().coords[1]))
    plt.scatter(*zip(*P), c=C)
    # plt.legend()
    #plt.scatter(*zip(*P))
    P = np.array(P)


    # plt.plot(P[:,0],P[:,1], 'b-', picker=5)
    #plt.plot([P[-1,0],P[0,0]],[P[-1,1],P[0,1]], 'b-', picker=5)
    #plt.plot(P[:,0],P[:,1], 'b')
    plt.plot(DEPOT_COORDS[0], DEPOT_COORDS[1], '*g')
    
    i = 0
    
    for route in routes:
        for edge in route.edges():
            vertices = edge.vertices()
            try:
                x_values = [vertices[0].element().coords[0], vertices[1].element().coords[0]]
                y_values = [vertices[0].element().coords[1], vertices[1].element().coords[1]]
            except AttributeError:
                x_values = [vertices[0].element()[0], vertices[1].element()[0]]
                y_values = [vertices[0].element()[1], vertices[1].element()[1]]
            plt.plot(x_values, y_values, colors[i % 7])
              
        i += 1


        #print(vertices[0].element().id)
        #plt.plot(vertices[0].element().coords, vertices[1].element().coords)
    #plt.axis('off')


    
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


if __name__ == '__main__':

    with open('orders.pkl', 'rb') as input:
        orders = pickle.load(input)

    routes = grasp(orders)

    '''
    for route in routes:
        for vertex in route.vertices():
            order = vertex.element()
            order.uid = random.randint(0,1000)
            if isinstance(order.scheduled_time, str):
                order.scheduled_time = START_TIME
            order.slack = order.scheduled_time - (START_TIME + datetime.timedelta(minutes=round(order.distance / SPEED)))
    '''
    '''
    orders = list()
    for route in routes:
        for order in route.vertices():
            orders.append(order.element())
    '''
    #routes = set_id_by_position(routes)
    '''
    with open("distances.txt", "a") as f:
        f.write("%f\n" % get_overall_distance(routes))
    plot_routes(routes, "Base")
    plt.show()

    local_search_routes, local_search_actioned = local_search(routes)
    plot_routes(local_search_routes, "Local Search")
    # plot_routes(two_opt_route_improve(local_search_routes), "two-opt again")
    # a = input("Press any key to continue")
    final_routes = two_opt_ls_iterator(local_search_routes, 3)
    with open("distances.txt", "a") as f:
        f.write("%f\n" % get_overall_distance(final_routes))
    plot_routes(final_routes, "Multiple Local Searches and 2-Opt Improvements")

    plt.show()
    '''