from Order import Order
from math import sqrt
from random import random, randrange
from operator import itemgetter, attrgetter
from grahamscan import GrahamScan
from Graph import Graph, Vertex
from Config import Config
import matplotlib.pyplot as plt
import random, operator, time, copy, pickle, datetime, math, sys
import numpy as np
from collections import Counter

#from py2opt.routefinder import RouteFinder
from routefinder.py2opt.py2opt.routefinder import RouteFinder

DEPOT_COORDS = (150, 150)
START_TIME = datetime.timedelta(hours=9)
MAX_NUMBER_OF_ROUTES = 6
SPEED = 9
NUMBER_OF_ORDERS = Config.NUMBER_OF_ORDERS
CONVERGENCE_COUNTER = 3

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

def order_is_reachable(route, order, flip_direction=False):
    total_time = 0

    edges = route.edges_in_order()
    if flip_direction:
        edges.reverse()

    #print('-- Order: %i' % order.order.id)
    #print(edges)
    
    for edge in edges:
        #print(edge)
        start = edge.start()
        end = edge.end()

        total_time += route.get_distance_between_vertices(start, end) / SPEED

        if start == order or end == order:
            break

    total_time = datetime.timedelta(minutes=total_time)
    
    # store if it is reachable or not
    reachable = START_TIME + total_time < order.element().scheduled_time
    '''
    if not reachable:
        print('\n########')
        print(START_TIME + total_time)
        print(order.element().scheduled_time)
        print('########\n')
    '''
    return reachable

def route_is_feasible(route):
    #print('\n########')
    #print('num vertices %i' % route.num_vertices())
    for vertex in route.vertices()[1:]:
        if not order_is_reachable(route, vertex):
            for vertex in route.vertices()[1:]:
                if not order_is_reachable(route, vertex, True):
                    '''print('\nRoute\n')
                    print(route)
                    print('not reachable')
                    print(vertex)
                    print('\n########')'''
                    return False
    #print('\n########')
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
                try:
                    min_distance_to_j = get_distance_between_vertices(seed.coords, j.coords)
                except Exception as e:
                    print(e)
                    print('seed')
                    print(seed)
                    print('j')
                    print(j)
                    time.sleep(10)
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

    if not routes_are_feasible(routes):
        sys.exit('\n# Caught Errror #\nCourier speed is too low to reach seeds within their time windows.\nIncrease courier speed to create seeds.\n')
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


def main_routing(unscheduled_orders, S, routes):

    # find the min insertion cost and the best route
    print('\n######\nMain Routing\n######\n')

    while(len(unscheduled_orders) > 0):
        if not routes_are_feasible(routes):
            print('not feasible')
            for route in routes:
                print(route)
            return routes
        #print(len(unscheduled_orders))
        order_rcps = list()
        for order in unscheduled_orders:
            route_cost_penalty = None
            min_cost_for_routes = dict()

            for route in routes:
                min_cost_for_route = None

                for edge in route.edges():
                    
                    start = edge.start()
                    end = edge.end()
                    order_v = route.add_vertex(order)

                    # add order to route and check if route is still fully feasible
                    route.add_vertex_between_vertices(order_v, start, end)
                    if route_is_feasible(route):

                        d3 = route.get_distance_between_vertices(start, end)
                        d1 = get_distance_between_vertices(start.order.coords, order.coords)
                        d2 = get_distance_between_vertices(order.coords, end.order.coords)

                        cost = d1 + d2 - d3
                        
                        # get the optimum route and its associated cost.
                        if route_cost_penalty == None or cost < route_cost_penalty[1]:
                            route_cost_penalty = [route, cost, 0, edge]

                        # get minimum cost for each route. To be used later to calculating penalty.
                        if min_cost_for_route == None or cost < min_cost_for_route:
                            min_cost_for_route = cost

                    # remove order to preserve routes original state
                    route.remove_vertex_and_repair(order_v)

                # if the order is not insertable in a route, set it to a very high value
                if min_cost_for_route == None:
                    min_cost_for_route = 1000

                # all edges have been traversed. Record the lowest insertion cost for this order on this route.
                min_cost_for_routes[route] = min_cost_for_route

            # best route for given order found.
            # now caculate the penalty using the best cost for other routes
            # penalty is the sum of the best costs routes besides it's optimum insertion route.
            if route_cost_penalty:
                for route, min_cost_for_route in min_cost_for_routes.items():
                    # TypeError: 'NoneType' object is not subscriptable
                    # route is not initialised because no feasibility
                    if route != route_cost_penalty[0]:
                        try:
                            route_cost_penalty[2] += min_cost_for_route
                        except Exception as e:
                            print(e)
                            print('\nDICT')
                            print(min_cost_for_routes)
                            print('\nItem in iter')
                            print(min_cost_for_route)
                            print('\nRoute in iter')
                            print(route)
                            print('\nroute_cost_penalty[0]')
                            print(route_cost_penalty[0])
                            time.sleep(5)
            # cannot be feasibly inserted into any route
            # therefore, set penalty to infinity.
            else:
                route_cost_penalty = [None, None, math.inf]

            # add to list of all order,root cost penalties
            order_rcps.append([order, route_cost_penalty])

        # get largest penalty orders
        largest_penalty_orders = sorted(order_rcps, key=lambda x: x[1][2], reverse=True)
        added_order_rcp = largest_penalty_orders[0]
        
        try:
            if added_order_rcp[1][2] != math.inf:
                # now route this order
                added_order = routes[routes.index(added_order_rcp[1][0])].add_vertex(added_order_rcp[0])
            # order cannot be added to any existing routes. Create a new dedicated route.
            else:
                g = Graph(99)
                depot = g.add_vertex(Order(0, 'ray', DEPOT_COORDS, START_TIME, 0))
                added_order = g.add_vertex(added_order_rcp[0])
                g.add_edge(depot, added_order)

                if not route_is_feasible(g):
                    sys.exit('\n# Caught Errror #\nCourier speed is too low to reach some vertex on a direct path.\nIncrease courier speed to allow feasibility.\n')
                else:
                    routes.append(g)

        except Exception as e:
            print('Failed to add vertex')
            print(e)
            time.sleep(4)

        # now connect the vertex with an edge 
        try:
            if added_order_rcp[1][2] != math.inf:
                w1 = added_order_rcp[1][3].start()
                w2 = added_order_rcp[1][3].end()
                added = routes[routes.index(added_order_rcp[1][0])].add_vertex_between_vertices(added_order, w1, w2)

        except Exception as e:
            print('Failed to add edge between vertices')
            print(e)
            time.sleep(4)
        
        try:
            unscheduled_orders.remove(added_order_rcp[0])
        except Exception as e:
            print(e)
            time.sleep(4)

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

def two_opt_route_improve_2(routes):

    improved_routes = list()
    for i in range(len(routes)):

        while True:
            print(i)
            dist_mat = get_distance_matrix(routes[i])
            route_finder = RouteFinder(dist_mat, routes[i].vertices(), iterations=10)
            best_distance, best_route = route_finder.solve()

            improved_route = Graph('4')
            time.sleep(4)

            for start, end in zip(best_route, best_route[1:]):

                v1 = improved_route.add_existing_vertex(routes[i].get_vertex(start.element()))
                v2 = improved_route.add_existing_vertex(routes[i].get_vertex(end.element()))

                improved_route.add_edge(v1, v2)
        
            improved_route.complete_route()

            if route_is_feasible(improved_route):
                print('infeasible')
                improved_routes.append(improved_route)
                #time.sleep(3)
                break
            else:
                print('infeasible')
                pass
    
    return set_id_by_position(improved_routes)


def cost_of_break(order, route):

    original_edges = route.get_edges(order)

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




def local_search_tw_shuffle_iterator(routes):

    open('num_vertices.txt', 'w').close()
    num_vertices = 0
    for route in routes:
        num_vertices += route.num_vertices() - 1
    with open("num_vertices.txt", "a") as f:
        f.write("Initial: %i\n" % num_vertices)
        f.write("Initial: %i\n" % get_overall_distance(routes))

    timeout = time.time() + 100
    best_distance = get_overall_distance(routes)
    convergence_counter = 0
    while True:
        # Local Search
        routes = local_search_tw(routes)[0]

        plot_routes(routes, "Local Search w/ TW")
        num_vertices = 0
        for route in routes:
            num_vertices += route.num_vertices() - 1

        with open("num_vertices.txt", "a") as f:
            f.write("LS\n%i\n" % num_vertices)
            f.write("%i\n" % get_overall_distance(routes))

        if num_vertices != NUMBER_OF_ORDERS:
            print('Local Search has lost vertices')
            time.sleep(3)
            return routes
        if not routes_are_feasible(routes):
            print('Local Search has caused infeasibility')
            time.sleep(3)
            return routes

        # TW Shuffle        
        routes, failed_vertices = tw_shuffle(routes)
        num_vertices = 0
        for route in routes:
            num_vertices += route.num_vertices() - 1

        with open("num_vertices.txt", "a") as f:
            f.write("Shuffle\n%i\n" % num_vertices)
            f.write("%i\n" % get_overall_distance(routes))
            f.write(str(failed_vertices) + "\n")
            f.write("Convergence Counter: " + str(convergence_counter) + "\n")

        if num_vertices != NUMBER_OF_ORDERS:
            print('TW Shuffle has lost vertices')
            print(num_vertices)
            print(failed_vertices)
            time.sleep(3)
            return routes

        if not routes_are_feasible(routes):
            print('TW Shuffle has caused infeasibility')
            time.sleep(3)
            return routes
        
        if get_overall_distance(routes) == best_distance:
            convergence_counter += 1
        elif get_overall_distance(routes) < best_distance:
            best_distance = get_overall_distance(routes)
            convergence_counter = 0

        if convergence_counter >= CONVERGENCE_COUNTER:
            break

    best_distance = get_overall_distance(routes)

    return routes


def local_search_tw_shuffle_iterator2(routes):

    open('num_vertices.txt', 'w').close()
    num_vertices = 0
    for route in routes:
        num_vertices += route.num_vertices() - 1
    with open("num_vertices.txt", "a") as f:
        f.write("Initial: %i\n" % num_vertices)
        f.write("Initial: %i\n" % get_overall_distance(routes))

    timeout = time.time() + 100
    best_distance = get_overall_distance(routes)
    convergence_counter = 0
    while True:
        # Local Search
        routes = local_search_tw(routes)[0]

        plot_routes(routes, "Local Search w/ TW")
        num_vertices = 0
        for route in routes:
            num_vertices += route.num_vertices() - 1

        with open("num_vertices.txt", "a") as f:
            f.write("LS\n%i\n" % num_vertices)
            f.write("%i\n" % get_overall_distance(routes))

        if num_vertices != NUMBER_OF_ORDERS:
            print('Local Search has lost vertices')
            time.sleep(3)
            return routes
        if not routes_are_feasible(routes):
            print('Local Search has caused infeasibility')
            time.sleep(3)
            return routes

        if get_overall_distance(routes) == best_distance:
            convergence_counter += 1
        elif get_overall_distance(routes) < best_distance:
            best_distance = get_overall_distance(routes)
            convergence_counter = 0

        if convergence_counter >= CONVERGENCE_COUNTER:
            break

    with open("num_vertices.txt", "a") as f:
            f.write("\n\n######################\n\n")


    timeout = time.time() + 100
    best_distance = get_overall_distance(routes)
    convergence_counter = 0
    while True:

        # TW Shuffle        
        routes, failed_vertices = tw_shuffle(routes)
        num_vertices = 0
        for route in routes:
            num_vertices += route.num_vertices() - 1

        with open("num_vertices.txt", "a") as f:
            f.write("Shuffle\n%i\n" % num_vertices)
            f.write("%i\n" % get_overall_distance(routes))
            f.write(str(failed_vertices) + "\n")
            f.write("Convergence Counter: " + str(convergence_counter) + "\n")

        if num_vertices != NUMBER_OF_ORDERS:
            print('TW Shuffle has lost vertices')
            print(num_vertices)
            print(failed_vertices)
            time.sleep(3)
            return routes

        if not routes_are_feasible(routes):
            print('TW Shuffle has caused infeasibility')
            time.sleep(3)
            return routes

        if get_overall_distance(routes) == best_distance:
            convergence_counter += 1
        elif get_overall_distance(routes) < best_distance:
            best_distance = get_overall_distance(routes)
            convergence_counter = 0

        if convergence_counter >= CONVERGENCE_COUNTER:
            break

        
        

    best_distance = get_overall_distance(routes)

    return routes

def tw_shuffle(routes):

    failed_vertices = list()
    improved_routes = list()
    for route in routes:
        if route.num_vertices() <= 2:
            improved_routes.append(route)
        else:
            
            failed_vertices_for_route = list()

            # create a new improved route with abritrary ID
            # this improved route will be used to record changes for all vertices for this given route.
            improved_route = Graph('4')

            shuffled_vertices = route.vertices()[1:]
            random.shuffle(shuffled_vertices)
            if len(shuffled_vertices) != len(route.vertices()[1:]):
                print(route.vertices()[1:])
                print(shuffled_vertices)
                time.sleep(6)

            # add the depot to the start of the improved route
            d0 = improved_route.add_existing_vertex(route.vertices()[0])
            # add the first vertex from the shuffled vertices to begin
            d1 = improved_route.add_existing_vertex(shuffled_vertices.pop(0))
            # connect the first vertex to the depot
            improved_route.add_edge(d0, d1)
            
            # continue until all shuffled vertices have been assigned.
            for vertex in shuffled_vertices:

                # create a list to record all feasible routes for this vertex.
                feasible_routes = list()
                # iterate through all edges to find the best position for this vertex
                for edge in improved_route.edges():
                    
                    start = edge.start()
                    end = edge.end()
                    improved_route.add_vertex_between_vertices(vertex, start, end)
                    '''
                    print('$$$$ ORIGINAL $$$$\n')
                    print(route)
                    #print('Distance: %f' % get_route_distance(improved_other_route))
                
                    print('\n----- Origin -----')
                    print('Order: %i' % (vertex.element().id))
                    print('------------------')

                    print('\n----- Destination -----')
                    print('Order: %i <-> %i' % (start.element().id, end.element().id))
                    print('-----------------------\n')
                    '''
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
                    print('FEASIBLE ROUTES')
                    print(feasible_routes)
                    
                    print(e)
                    failed_vertices_for_route.append(vertex)
                    #time.sleep(5)
                    route.complete_route()
                    improved_routes.append(route)
                    # break out of edges for loop. Iterate to next route
                    break

            # complete the route
            improved_route.complete_route()
            route.complete_route()
            # record the finalised improved route
            if improved_route.num_vertices() != route.num_vertices() or get_route_distance(improved_route) > get_route_distance(route):
                print('\nOriginal')
                print(route)
                print(route.vertices()[1:])
                print('\nImproved')
                print(improved_route)
                print(improved_route.vertices()[1:])
                #print('Original feasible')
                #print(route_is_feasible(route))
                if improved_route.num_vertices() != route.num_vertices():
                    print('LOST VERTICES\n')
                elif get_route_distance(improved_route) > get_route_distance(route):
                    
                    print('DISTANCE INCREASE\n')
                    print('Original %f' % get_route_distance(route))
                    print('Improved %f' % get_route_distance(improved_route))

                time.sleep(5)
                improved_routes.append(route)
                continue
            else:
                improved_routes.append(improved_route)
                continue

            
            failed_vertices.append(failed_vertices_for_route)
    for vertex_set in failed_vertices:
        if vertex_set:
            print('###')
            print(vertex_set)
            print('###')

    if len(improved_routes) > 0:
        for i in range(len(improved_routes)):
            try:
                if improved_routes[i].num_vertices() != routes[i].num_vertices() \
                    or get_route_distance(improved_routes[i]) > get_route_distance(routes[i]):
                    '''
                    print('BAD CAUGHT')
                    print('Original')
                    print(routes[i])
                    print('Improved')
                    print(improved_routes[i])
                    '''
                    improved_routes[i] = routes[i]
                    #time.sleep(5)
            except Exception as e:
                print(e)
                time.sleep(4)
                        
    return improved_routes, failed_vertices


def local_search_tw(routes):

    if not routes_are_feasible(routes):
        sys.exit('\n# local_search_tw given bad input. #\n')

    local_search_actioned = False

    shuffled_routes = routes
    random.shuffle(shuffled_routes)

    if not routes_are_feasible(shuffled_routes):
        sys.exit('\nBad from shuffle\n')

    for i, origin_route in enumerate(shuffled_routes): 
        
        shuffled_vertices = origin_route.vertices()[1:]
        random.shuffle(shuffled_vertices)

        if not routes_are_feasible(shuffled_routes):
            sys.exit('\n1\n')
        for order in shuffled_vertices:
            if not routes_are_feasible(shuffled_routes):
                print(shuffled_vertices)
                print(order)
                print('\ns\n')
                return (shuffled_routes, local_search_actioned)
                sys.exit('\ns\n')
                
            other_routes = shuffled_routes.copy()
            if not routes_are_feasible(shuffled_routes):
                sys.exit('\nt\n')
            other_routes.pop(shuffled_routes.index(origin_route))
            if not routes_are_feasible(shuffled_routes):
                sys.exit('\nz\n')
            if not routes_are_feasible(shuffled_routes):
                sys.exit('\n?\n')
            best_new_position_route = None
            # iterate through all the routes except for the origin route.
            for j in range(len(other_routes)):
                # improved_other_route is the destination route being considered.
                improved_other_route = copy.deepcopy(other_routes[j])
                if not routes_are_feasible(shuffled_routes):
                    sys.exit('\n*\n')
                # find best location for all edges in destination route
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
                    print('Route: %s, Order: %i <-> %i' % (colors[shuffled_routes.index(other_routes[j]) % 7], start.element().id, end.element().id))
                    print('-----------------------\n')
                    
                    if not routes_are_feasible(shuffled_routes):
                        sys.exit('\n2\n')
                    improved_routes = shuffled_routes.copy()
                    improved_origin_route = copy.deepcopy(improved_routes[i])

                    if origin_route.degree(order) > 2:
                        # time.sleep(1)
                        print('TOO MANY EDGES')

                    #print('$$$$ Origin Route $$$$\n')
                    #print(origin_route)

                    # add the vertex to the destination route between the edge vertices being considered
                    improved_other_route.add_vertex_between_vertices(order, start, end)
                    if not routes_are_feasible(shuffled_routes):
                        sys.exit('\n2.5\n')

                    '''
                    print('\n$$$$ CHANGES $$$$')
                    print(improved_other_route)
                    print('Distance: %f' % get_route_distance(improved_other_route))
                    print('\n')
                    '''
                    try:
                        # print('Best Route Distance: %f' % get_route_distance(best_new_position_route[0]))
                        print('Best Route Distance: %f' % best_new_position_route[4])
                        print('This Route Distance: %f' % get_route_distance(improved_other_route))
                    except Exception as e: 
                        pass


                    # find the new position with the lowest cost
                    try:
                        if route_is_feasible(improved_other_route) and (best_new_position_route == None or \
                        route_distance_difference(improved_other_route, other_routes[j]) < route_distance_difference(best_new_position_route[0], other_routes[best_new_position_route[1]])):
                            
                            best_new_position_route = [copy.deepcopy(improved_other_route), j, start.element().uid, end.element().uid]
                        
                        # make reparations 
                        improved_other_route.remove_vertex_and_repair(order)
                        #print(improved_other_route)
                        #print('Distance: %f' % get_route_distance(improved_other_route))
                        #print('$$$$ END $$$$\n')
                    except Exception as e:
                        print(e)
                        plot_routes(routes, "error on feasibility")
                        time.sleep(10)
                        plt.show()


            if not routes_are_feasible(shuffled_routes):
                        sys.exit('\n5\n')
                        pass
            if best_new_position_route == None:
                break
            # check if dest route distance + cost of break in origin route is better than old dest route distance
            if best_new_position_route != None and route_is_feasible(best_new_position_route[0]) and route_is_feasible(origin_route) and ((cost_of_break(order, origin_route) + get_route_distance(best_new_position_route[0])) \
                < (get_route_distance(shuffled_routes[shuffled_routes.index(other_routes[best_new_position_route[1]])]))):

                local_search_actioned = True
                try:
                    if not routes_are_feasible(shuffled_routes):
                        sys.exit('\n# start. #\n')
                        pass
                    shuffled_routes[i].remove_vertex_and_repair(order)
                    shuffled_vertices.remove(order)

                    #-------------------------------
                    # get the index for the destination route in shuffled_routes.
                    r_index = shuffled_routes.index(other_routes[best_new_position_route[1]])
                    
                    start = shuffled_routes[r_index].get_vertex_by_uid(best_new_position_route[2])
                    end = shuffled_routes[r_index].get_vertex_by_uid(best_new_position_route[3])
                    
                    # Add order to destination route
                    shuffled_routes[r_index].add_vertex_between_vertices(order, start, end)
                    if not routes_are_feasible(shuffled_routes):
                        #sys.exit('\n# local_search_tw given bad input. #\n')
                        pass
                    
                except Exception as e:
                    print(e)
                    print('ERROR')
                    time.sleep(5)
                    plot_routes(shuffled_routes, "error")
                    plt.show()
                  
            else:
                if not routes_are_feasible(shuffled_routes):
                    sys.exit('\nBAD ELSE\n')
                '''
                print('fail')
                print(get_route_distance(origin_route) + cost_of_break(order, origin_route) + \
                get_route_distance(best_new_position_route[0]))
                print((get_route_distance(origin_route) + \
                get_route_distance(routes[routes.index(other_routes[best_new_position_route[1]])])))
                print(cost_of_break(order, origin_route))
                '''
    return (shuffled_routes, local_search_actioned)

def grasp(orders, graph_results=True):

    L_points, L = GrahamScan(orders)
    L_comp = list(set(orders) - set(L))
    #plot_convex_hull(orders, L_points, '.r')
    S, L, L_comp, routes = route_initialization(L, L_comp)
    unscheduled_orders = L + L_comp    

    routes = main_routing(unscheduled_orders, S, routes)

    if not routes_are_feasible(routes):
        sys.exit('infeasible routes')
        
    if graph_results:
        plot_routes(routes, "main routing 1")
   
    routes = local_search_tw_shuffle_iterator2(routes)

    if graph_results:
        plot_routes(routes, "Iterator")
        plt.show()
    return routes
   

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
    route_directed_edges = dict()
    for route in routes:
        if route.num_vertices() <= 1:
            break

        edges = route.edges_in_order()
        for vertex in route.vertices()[1:]:
            if not order_is_reachable(route, vertex):
                for vertex in route.vertices()[1:]:
                    if not order_is_reachable(route, vertex, True):
                        print('UNREACHABLE')
                        sys.exit('UNREACHABLE ROUTE')
                edges.reverse()
                break
        route_directed_edges[route] = edges
    
    for route, edges in route_directed_edges.items():
        print('\n-----------------')
        print(colors[routes.index(route) % 7])
        print('-----------------\n')

        print([edge.vertices() for edge in edges])
        print('\n')
    
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

    with open('../saved_routes/routes_13.pkl', 'rb') as input:
        routes_original = pickle.load(input)
    routes_original = [route for route in routes_original if route.num_edges() > 0]

    plot_routes(routes_original, "Base")

    plt.show()
    original_num_vertices=0

    for route in routes_original:
        original_num_vertices += route.num_vertices() - 1
    
    #routes = two_opt_route_improve_2(routes_original)
    #print(routes)
    #time.sleep(30)
    routes = local_search_tw_shuffle_iterator(routes_original)

    num_vertices=0
    for route in routes:
        num_vertices += route.num_vertices() - 1
    print('ORIG NUM VERTICES')
    print(original_num_vertices)
    print('FEASIBLE')
    print(routes_are_feasible(routes_original))

    print('NUM VERTICES')
    print(num_vertices)
    
    print('FEASIBLE')
    print(routes_are_feasible(routes))
    print('******************')
    plot_routes(routes, "Iterator")
    plt.show()


    routes_original_ids = list()
    routes_ids = list()

    for route in routes_original:
        for vertex in route.vertices():
            routes_original_ids.append(vertex.order.uid)

    for route in routes:
        for vertex in route.vertices():
            routes_ids.append(vertex.order.uid)

    missing_uids = list(set(routes_original_ids) - set(routes_ids))
    item = missing_uids[0]

    for uid in missing_uids:
        if uid in routes_original_ids:
            for route in routes_original:
                if route.get_vertex_by_uid(uid):
                    print(colors[routes_original.index(route)])
                    print(route.get_vertex_by_uid(uid))
        else:
            for route in routes:
                if route.get_vertex_by_uid(uid):
                    print(route.get_vertex_by_uid(uid))
    #print(routes_original_ids)

    

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