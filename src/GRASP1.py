from Order import Order
from math import sqrt
from random import random, randrange
from operator import itemgetter, attrgetter
from Route import Route
from read_config import read_config
import matplotlib.pyplot as plt
import random, operator, time, copy, pickle, datetime, math, sys
import numpy as np
from collections import Counter

#from py2opt.routefinder import RouteFinder
#from VRPTW.routefinder.py2opt.py2opt.routefinder import RouteFinder

DEPOT_COORDS = (150, 150)
START_TIME = datetime.timedelta(hours=9)
#MAX_NUMBER_OF_ROUTES = 6
#SPEED = 9
#NUMBER_OF_ORDERS = Config.NUMBER_OF_ORDERS
#CONVERGENCE_COUNTER = 3

config_vars = read_config()
for key,val in config_vars.items():
    exec(key + '=val')


local_search_actioned = False
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']


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

def clean_routes(routes):

    # remove any empty routes
    routes = [route for route in routes if route.num_orders() > 1]

    for route in routes:
        route.complete_route()

    return routes

def distance_to_order(route, order):

    total_distance = 0
    edges = route.edges_in_order()

    for edge in edges:

        start = edge.start()
        end = edge.end()

        total_distance += route.get_distance_between_orders(start, end)

        if start == order or end == order:
            break

    return total_distance


def order_is_reachable(route, order, flip_direction=False):
    total_time = 0

    edges = route.edges_in_order_undirected()
    if flip_direction:
        edges.reverse()

    #print('-- Order: %i' % order.order.id)
    #print(edges)
    
    for edge in edges:
        #print(edge)
        start = edge.start()
        end = edge.end()

        total_time += route.get_distance_between_orders(start, end) / SPEED

        if start == order or end == order:
            break

    total_time = datetime.timedelta(minutes=total_time)
    
    # store if it is reachable or not
    reachable = START_TIME + total_time < order.scheduled_time
    '''
    print('\n########')
    print(SPEED)
    print('\n')
    print(START_TIME + total_time)
    print(order.element().scheduled_time)
    print(reachable)
    print('########\n')
    time.sleep(3)
    '''

    return reachable


def routes_are_feasible(routes):
    routes = clean_routes(routes)
    for route in routes:
        if not route.is_feasible():
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

def get_distance_between_orders(o1, o2):

    return sqrt((o1.coords[0] - o2.coords[0])**2 + (o1.coords[1] - o2.coords[1])**2)

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

'''
    Input: List of orders
    Output: List of routes with one order(seed) in each, list of unscheduled orders

    S = empty list
    First_seed = furthest order from the depot
    Add first seed to list of seeds S. Remove first seed from list of orders

    While number of seeds is less than the maximum number of routes:

        Find the order that maximises the sum of distances to the existing seeds in S
        Add the order to the list of seeds S. Remove the order from list of orders

    R = empty list
    For each seed in the list of seeds S:
        Create a new route
        Add the seed to the route and connect it to the depot
        Add this route to the list of routes R

    Return: R, Orders
'''
def route_initialization(orders):

    S = list()

    first_seed = max(orders, key=attrgetter('distance_to_depot'))
    S.append(first_seed)

    orders.remove(first_seed)

    while len(S) < MAX_NUMBER_OF_ROUTES:

        # find order that maximises sum of distances from existing seeds in S
        new_seed = (None, 0)
        for order in orders:
            accum_distance = 0
            for seed in S:
                accum_distance += get_distance_between_orders(seed, order)
            if accum_distance > new_seed[1]:
                new_seed = (order, accum_distance)
        
        new_seed = new_seed[0]

        orders.remove(new_seed)
        new_seed.seed = True
        S.append(new_seed)


    routes = list()
    i = 0
    for seed in S:
        g = Route()
        depot = g.add_order(Order(0, 'ray', DEPOT_COORDS, START_TIME, 0))
        seed = g.add_order(seed)
        g.add_edge(depot, seed)
        routes.append(g)
        #routes[len(routes) - 1].add_vertex(Order(0, 'ray', DEPOT_COORDS, '00:00', '00:00', 0))
        i += 1

    if not routes_are_feasible(routes):
        sys.exit('\n# Caught Errror #\nCourier speed is too low to reach seeds within their time windows.\nIncrease courier speed to create seeds.\n')


    return [routes, orders]

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

'''
Input: List of routes, list of unscheduled orders
Output: List of routes containing all orders

While |unscheduled orders| > 0:

    For each unscheduled order in the list unscheduled orders:
        For each route:
            Find the minimum insertion cost for the order in this route

        Find the route with the lowest minimum insertion cost
        If the order can be inserted:
            Calculate the penalty cost for this order
        Else:
            Set the penalty value to infinity for this order

    Select the customer with the largest penalty value
    If the order has a feasible best insertion location:
        Insert this order into its insertion location
    Else if the order does not habe a feasible best insertion location:
        Create a new dedicated route for this order
        Add the order to the new route

Return: Complete Routes
'''
def main_routing(routes, unscheduled_orders):

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
                    
                    o1 = edge.start()
                    o2 = edge.end()
                    order = route.add_order(order)
                    distance_to_o2 = distance_to_order(route, o2)
                    # add order to route and check if route is still fully feasible
                    route.add_order_between_orders(order, o1, o2)
                    if route.is_feasible():

                        d3 = route.get_distance_between_orders(o1, o2)
                        d1 = get_distance_between_orders(o1, order)
                        d2 = get_distance_between_orders(order, o2)

                        cost = (d1 + d2 - d3) + (distance_to_order(route, o2) - distance_to_o2)
                        
                        # get the optimum route and its associated cost.
                        if route_cost_penalty == None or cost < route_cost_penalty[1]:
                            route_cost_penalty = [route, cost, 0, edge]

                        # get minimum cost for each route. To be used later to calculating penalty.
                        if min_cost_for_route == None or cost < min_cost_for_route:
                            min_cost_for_route = cost

                    # remove order to preserve routes original state
                    route.remove_order_and_repair(order)

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
                added_order = routes[routes.index(added_order_rcp[1][0])].add_order(added_order_rcp[0])
            # order cannot be added to any existing routes. Create a new dedicated route.
            else:
                g = Route(99)
                depot = g.add_order(Order(0, 'ray', DEPOT_COORDS, START_TIME, 0))
                added_order = g.add_order(added_order_rcp[0])
                g.add_edge(depot, added_order)

                if not g.is_feasible():
                    sys.exit('\n# Caught Errror #\nCourier speed is too low to reach some order on a direct path.\nIncrease courier speed to allow feasibility.\n')
                else:
                    routes.append(g)

        except Exception as e:
            print('Failed to add order')
            print(e)
            time.sleep(4)

        # now connect the vertex with an edge 
        try:
            if added_order_rcp[1][2] != math.inf:

                w1 = added_order_rcp[1][3].start()
                w2 = added_order_rcp[1][3].end()
                added = routes[routes.index(added_order_rcp[1][0])].add_order_between_orders(added_order, w1, w2)

        except Exception as e:
            print('Failed to add edge between orders')
            print(e)
            time.sleep(4)
        
        try:
            unscheduled_orders.remove(added_order_rcp[0])
        except Exception as e:
            print(e)
            time.sleep(4)

    routes = clean_routes(routes)

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

            if improved_route.is_feasible():
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

        o1 = original_edges[0].opposite(order)
        o2 = original_edges[1].opposite(order)

        
        d1 = route.get_distance_between_orders(o1, order)
        d2 = route.get_distance_between_orders(order, o2)

        d3 = route.get_distance_between_orders(o1, o2)
    else:
        o1 = original_edges[0].opposite(order)
        d1 = route.get_distance_between_orders(o1, order)
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




def local_search_tw_shuffle_iterator(routes, graph_results=True):

    open('num_orders.txt', 'w').close()
    num_orders = 0
    for route in routes:
        num_orders += route.num_orders() - 1
    with open("num_orders.txt", "a") as f:
        f.write("Initial: %i\n" % num_orders)
        f.write("Initial: %i\n" % get_overall_distance(routes))

    timeout = time.time() + 100
    best_distance = get_overall_distance(routes)
    convergence_counter = 0
    iteration_counter = 0
    while True:
        
        # Local Search
        routes = local_search(routes)[0]
        num_orders = 0
        for route in routes:
            num_orders += route.num_orders() - 1

        with open("num_orders.txt", "a") as f:
            f.write("LS\n%i\n" % num_orders)
            f.write("%i\n" % get_overall_distance(routes))

        if num_orders != NUMBER_OF_ORDERS:
            print('Local Search has lost vertices')
            time.sleep(3)
            return routes
        if not routes_are_feasible(routes):
            print('Local Search has caused infeasibility')
            time.sleep(3)
            return routes

        # TW Shuffle        
        routes, failed_vertices = tw_shuffle(routes)
        num_orders = 0
        for route in routes:
            num_orders += route.num_orders() - 1

        with open("num_orders.txt", "a") as f:
            f.write("Shuffle\n%i\n" % num_orders)
            f.write("%i\n" % get_overall_distance(routes))
            f.write(str(failed_vertices) + "\n")
            f.write("Convergence Counter: " + str(convergence_counter) + "\n")

        if num_orders != NUMBER_OF_ORDERS:
            print('TW Shuffle has lost vertices')
            print(num_orders)
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

            iteration_counter += 1
            if graph_results:
                plot_routes(routes, "Improvement Round %i" % iteration_counter)


        if convergence_counter >= CONVERGENCE_COUNTER:
            break

    best_distance = get_overall_distance(routes)

    return routes


def local_search_tw_shuffle_iterator2(routes):

    open('num_orders.txt', 'w').close()
    num_orders = 0
    for route in routes:
        num_orders += route.num_orders() - 1
    with open("num_orders.txt", "a") as f:
        f.write("Initial: %i\n" % num_orders)
        f.write("Initial: %i\n" % get_overall_distance(routes))

    timeout = time.time() + 100
    best_distance = get_overall_distance(routes)
    convergence_counter = 0
    while True:
        # Local Search
        routes = local_search(routes)[0]

        plot_routes(routes, "Local Search w/ TW")
        num_orders = 0
        for route in routes:
            num_orders += route.num_orders() - 1

        with open("num_orders.txt", "a") as f:
            f.write("LS\n%i\n" % num_orders)
            f.write("%i\n" % get_overall_distance(routes))

        if num_orders != NUMBER_OF_ORDERS:
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

    with open("num_orders.txt", "a") as f:
            f.write("\n\n######################\n\n")


    timeout = time.time() + 100
    best_distance = get_overall_distance(routes)
    convergence_counter = 0
    while True:

        # TW Shuffle        
        routes, failed_vertices = tw_shuffle(routes)
        num_orders = 0
        for route in routes:
            num_orders += route.num_orders() - 1

        with open("num_orders.txt", "a") as f:
            f.write("Shuffle\n%i\n" % num_orders)
            f.write("%i\n" % get_overall_distance(routes))
            f.write(str(failed_vertices) + "\n")
            f.write("Convergence Counter: " + str(convergence_counter) + "\n")

        if num_orders != NUMBER_OF_ORDERS:
            print('TW Shuffle has lost vertices')
            print(num_orders)
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
'''
Input: list of routes

Improved_routes = empty list

For each route in routes
    create a new blank route 'improved_route'
    make a copy of all of the routes vertices and shuffle it
    add the depot and the first vertex from the shuffle vertices to the new route
    connect these two vertices with an edge

    for each vertex in shuffled vertices
        find the best feasible location for the vertex in the partially built route 'improved_route'
        if feasible location found:
            add the vertex to improved_route
        if feasible location not found:
            break out of loop and preserve the original route

    add improved route to list 'improved routes'

Return: improved_routes
'''
def tw_shuffle(routes):

    failed_orders = list()
    improved_routes = list()
    for route in routes:
        if route.num_orders() <= 2:
            improved_routes.append(route)
        else:
            
            failed_orders_for_route = list()

            # create a new improved route with abritrary ID
            # this improved route will be used to record changes for all vertices for this given route.
            improved_route = Route()

            shuffled_orders = route.orders()[1:]
            random.shuffle(shuffled_orders)
            if len(shuffled_orders) != len(route.orders()[1:]):
                print(route.orders()[1:])
                print(shuffled_orders)
                time.sleep(6)

            # add the depot to the start of the improved route
            d0 = improved_route.add_order(route.orders()[0])
            # add the first vertex from the shuffled vertices to begin
            d1 = improved_route.add_order(shuffled_orders.pop(0))
            # connect the first vertex to the depot
            improved_route.add_edge(d0, d1)
            
            # continue until all shuffled vertices have been assigned.
            for order in shuffled_orders:

                # create a list to record all feasible routes for this vertex.
                feasible_routes = list()
                # iterate through all edges to find the best position for this vertex
                for edge in improved_route.edges():
                    
                    start = edge.start()
                    end = edge.end()
                    improved_route.add_order_between_orders(order, start, end)
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
                    if improved_route.is_feasible():
                        # add route to list of all feasible routes for this order
                        feasible_routes.append(copy.deepcopy(improved_route))
                    # undo changes
                    improved_route.remove_order_and_repair(order)
                    
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
                    failed_orders_for_route.append(order)
                    #time.sleep(5)
                    route.complete_route()
                    improved_routes.append(route)
                    # break out of edges for loop. Iterate to next route
                    break

            # complete the route
            improved_route.complete_route()
            route.complete_route()
            # record the finalised improved route
            if improved_route.num_orders() != route.num_orders() or get_route_distance(improved_route) > get_route_distance(route):
                print('\nOriginal')
                print(route)
                print(route.orders()[1:])
                print('\nImproved')
                print(improved_route)
                print(improved_route.orders()[1:])
                #print('Original feasible')
                if improved_route.num_orders() != route.num_orders():
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

            
            failed_orders.append(failed_orders_for_route)
    for order_set in failed_orders:
        if order_set:
            print('###')
            print(order_set)
            print('###')

    if len(improved_routes) > 0:
        for i in range(len(improved_routes)):
            try:
                if improved_routes[i].num_orders() != routes[i].num_orders() \
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
                        
    return improved_routes, failed_orders

'''
Input: List of routes, R
Output: List of routes

For each route r in R:
    For each order in r:
        For every possible insertion point p in neighbouring routes:
            Calculate the cost of inserting this order at p
            Monitor & Record the insertion point with the lowest insertion cost, b

        If moving the order to b lowers the overall routes distance:
            Move the order and update R
        Else:
            Preserve the orders location.

Return: R
'''

def local_search(routes):

    if not routes_are_feasible(routes):
        sys.exit('\n# local_search given bad input. #\n')

    local_search_actioned = False

    shuffled_routes = routes
    random.shuffle(shuffled_routes)

    if not routes_are_feasible(shuffled_routes):
        sys.exit('\nBad from shuffle\n')

    for i, origin_route in enumerate(shuffled_routes): 
        
        shuffled_orders = origin_route.orders()[1:]
        random.shuffle(shuffled_orders)

        if not routes_are_feasible(shuffled_routes):
            sys.exit('\n1\n')
        for order in shuffled_orders:
            if not routes_are_feasible(shuffled_routes):
                print(shuffled_orders)
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
                    print('Route: %s, Order: %i' % (colors[i % 7], order.id))
                    print('------------------')

                    print('\n----- Destination -----')
                    print('Route: %s, Order: %i <-> %i' % (colors[shuffled_routes.index(other_routes[j]) % 7], start.id, end.id))
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
                    improved_other_route.add_order_between_orders(order, start, end)
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
                        if improved_other_route.is_feasible() and (best_new_position_route == None or \
                        route_distance_difference(improved_other_route, other_routes[j]) < route_distance_difference(best_new_position_route[0], other_routes[best_new_position_route[1]])):
                            
                            best_new_position_route = [copy.deepcopy(improved_other_route), j, start.uid, end.uid]
                        
                        # make reparations 
                        improved_other_route.remove_order_and_repair(order)
                        #print(improved_other_route)
                        #print('Distance: %f' % get_route_distance(improved_other_route))
                        #print('$$$$ END $$$$\n')
                    except Exception as e:
                        print(e)
                        print("error on feasibility")
                        time.sleep(10)
                        plot_routes(routes, "error on feasibility")
                        time.sleep(10)
                        plt.show()


            if not routes_are_feasible(shuffled_routes):
                        sys.exit('\n5\n')
                        pass
            if best_new_position_route == None:
                break
            # check if dest route distance + cost of break in origin route is better than old dest route distance
            if best_new_position_route != None and best_new_position_route[0].is_feasible() and origin_route.is_feasible() and ((cost_of_break(order, origin_route) + get_route_distance(best_new_position_route[0])) \
                < (get_route_distance(shuffled_routes[shuffled_routes.index(other_routes[best_new_position_route[1]])]))):

                local_search_actioned = True
                try:
                    if not routes_are_feasible(shuffled_routes):
                        sys.exit('\n# start. #\n')
                        pass
                    shuffled_routes[i].remove_order_and_repair(order)
                    shuffled_orders.remove(order)

                    #-------------------------------
                    # get the index for the destination route in shuffled_routes.
                    r_index = shuffled_routes.index(other_routes[best_new_position_route[1]])
                    
                    start = shuffled_routes[r_index].get_order_by_uid(best_new_position_route[2])
                    end = shuffled_routes[r_index].get_order_by_uid(best_new_position_route[3])
                    
                    # Add order to destination route
                    shuffled_routes[r_index].add_order_between_orders(order, start, end)
                    if not routes_are_feasible(shuffled_routes):
                        #sys.exit('\n# local_search given bad input. #\n')
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

    
    # route initilization
    routes, unscheduled_orders = route_initialization(orders)    

    # main routing
    routes = main_routing(routes, unscheduled_orders)

    if not routes_are_feasible(routes):
        sys.exit('infeasible routes')
        
    if graph_results:
        plot_routes(routes, "Initial Route Construction")
   
    routes = local_search_tw_shuffle_iterator(routes, graph_results)

    routes = clean_routes(routes)

    return routes
   

def set_id_by_position(routes):
    for route in routes:
        for pos, order in enumerate(route.vertices()):
            order.id = pos
    
    return routes
    

def plot_routes(routes, title="untitled", labeled=True):


    fig = plt.figure()
    plt.suptitle(title, fontsize=20)

    P = list()
    #C = list()
    for route in routes:
        for order in route.orders():
            if not order.seed:
                P.append(order.coords)
                #C.append(np.cos(order.slack.total_seconds()))
            else:
                plt.plot(order.coords[0], order.coords[1], 'sk')
            
            if labeled:
                plt.annotate(order.id, (order.coords[0], order.coords[1]))
    plt.scatter(*zip(*P))
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
            orders = edge.orders()
            try:
                x_values = [orders[0].coords[0], orders[1].coords[0]]
                y_values = [orders[0].coords[1], orders[1].coords[1]]
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
        if route.num_orders() <= 1:
            break

        edges = route.edges_in_order_undirected()
        route_directed_edges[route] = edges
    
    for route, edges in route_directed_edges.items():
        print('\n-----------------')
        print(colors[routes.index(route) % 7])
        print('-----------------\n')

        print([edge.orders() for edge in edges])
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
    
    orders = create_orders(NUMBER_OF_ORDERS)
    print('22')
    print(orders)
    print(NUMBER_OF_ORDERS)

    start = time.time()
    routes_1 = grasp(orders, GRAPH_ROUTES)
    routes_1_time = round(time.time() - start, 3)
    routes_1_distance = round(get_overall_distance(routes_1), 3)
    print('Distance')
    print(routes_1_distance)

    if GRAPH_ROUTES:
        plot_routes(routes_1)
        plt.show()


    '''

    with open('../saved_routes/routes_13.pkl', 'rb') as input:
        routes_original = pickle.load(input)
    routes_original = [route for route in routes_original if route.num_edges() > 0]

    plot_routes(routes_original, "Base")

    plt.show()
    original_num_orders=0

    for route in routes_original:
        original_num_orders += route.num_orders() - 1
    
    #routes = two_opt_route_improve_2(routes_original)
    #print(routes)
    #time.sleep(30)
    routes = local_search_tw_shuffle_iterator(routes_original)

    num_orders=0
    for route in routes:
        num_orders += route.num_orders() - 1
    print('ORIG NUM VERTICES')
    print(original_num_orders)
    print('FEASIBLE')
    print(routes_are_feasible(routes_original))

    print('NUM VERTICES')
    print(num_orders)
    
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