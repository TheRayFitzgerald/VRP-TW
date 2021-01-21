from Van import Van
from Courier import Courier
from Order import Order
from math import sqrt
from random import random, randrange
from operator import itemgetter, attrgetter
from grahamscan import GrahamScan
from Graph import Graph, Vertex
import matplotlib.pyplot as plt
import random, operator, time, copy
import numpy as np
from collections import Counter

from py2opt.routefinder import RouteFinder

DEPOT_COORDS = (150, 150)
MAX_NUMBER_OF_ROUTES = 4
local_search_actioned = False

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

def get_route_distance(route):

    total_distance = 0

    for previous, current in zip(route.vertices(), route.vertices()[1:]):
        total_distance += get_distance_between_vertices(previous.element().coords, current.element().coords)

    return total_distance

def get_overall_distance(routes):

    overall_distance = 0

    for route in routes:
        overall_distance += get_route_distance(route)

    return overall_distance


def route_initialization(L, L_comp):

    # sort by shortest distance from depot
    L_comp.sort(key=lambda x: x.distance, reverse=True)

    S = list()
    S.append(max(L, key=attrgetter('distance')))

    # check this line with KB
    L.remove(max(L, key=attrgetter('distance')))

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
        g.add_vertex(Order(0, 'ray', DEPOT_COORDS, '00:00', '00:00', 0))
        g.add_vertex(seed, seed=True)
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

def main_routing(unscheduled_orders, S, routes):

    # find the min insertion cost and the best route

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

            # best route found. now caculate the penalty using the best cost for other routess
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
            routes[routes.index(added_order_rcp[1][0])].add_edge(added_order_rcp[1][3], added_order, 3)
            #routes[routes.index(added_order_rcp[1][0])].add_edge(routes[routes.index(added_order_rcp[1][0])].vertices()[len(routes[routes.index(added_order_rcp[1][0])].vertices()) - 3], added_order, 3)
        except Exception as e:
            print(e)
        

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

    for route in routes:

        route.add_existing_vertex(route.vertices()[0])
        route.add_edge(route.vertices()[-2], route.vertices()[-1], 0)


    return routes

# takes routes and utilises 2-opt to imrpove the routes
def two_opt_route_improve(routes):

    improved_routes = list()
    for i in range(len(routes)):

        dist_mat = get_distance_matrix(routes[i])
        route_finder = RouteFinder(dist_mat, routes[i].vertices(), iterations=5)
        best_distance, best_route = route_finder.solve()

        improved_route = Graph('4')

        for previous, current in zip(best_route, best_route[1:]):

            v1 = improved_route.add_existing_vertex(routes[i].get_vertex(previous.element()))
            v2 = improved_route.add_existing_vertex(routes[i].get_vertex(current.element()))

            improved_route.add_edge(v1, v2, get_distance_between_vertices(v1.element().coords, v2.element().coords))

        distance = get_distance_between_vertices(improved_route.vertices()[-1].element().coords, improved_route.vertices()[0].element().coords)
        improved_route.add_edge(improved_route.vertices()[-1], improved_route.vertices()[0], distance)
        improved_routes.append(improved_route)

    return improved_routes


def cost_of_break(order, route):

    original_distance = get_route_distance(route)
    route_copy = copy.deepcopy(route)

    original_edges = route.get_edges(order)

    v1 = original_edges[0].opposite(order)
    v2 = original_edges[1].opposite(order)

    d1 = get_distance_between_vertices(v1.element().coords, order.element().coords)
    d2 = get_distance_between_vertices(order.element().coords, v2.element().coords)

    d3 = get_distance_between_vertices(v1.element().coords, v2.element().coords)
  
    return d3 - (d1 + d2)

def local_search2(routes):

    '''
    Consider an order
    Find the best location for that order:
    Consider two connected orders in a separate route(do this for all order pairs in all other routes)
        insert the order between the two orders
        check if total route distance for the two routes has decreased
        yes:
            modify the two routes
        no:
            pass, iterate
    '''
    local_search_actioned = False
    t=[]
    g=[]
    for i in range(len(routes)):
        origin_route = routes[i]
        for order in origin_route.vertices()[1:]:
            best_new_position_route = None
            other_routes = routes.copy()
            other_routes.pop(routes.index(origin_route))

            # original best distance
            best_overall_distance = get_overall_distance(routes)
            best_routes_for_order = routes

            # magic
            for j in range(len(other_routes)):
                improved_other_route = copy.deepcopy(other_routes[j])
                for previous, current in zip(improved_other_route.vertices(), improved_other_route.vertices()[1:]):
                    improved_routes = routes.copy()
                    improved_origin_route = copy.deepcopy(improved_routes[i])

                    if origin_route.degree(order) > 2:
                        time.sleep(1)
                        print('TOO MANY EDGES')

                    print('POSITION')
                    print(origin_route.get_vertex_position(order))

                    #origin_route_test_copy = copy.deepcopy(origin_route)
                    original_edges = origin_route.get_edges(order)
                    print('EDGES')
                    print(original_edges)

                    v1 = original_edges[0].opposite(order)
                    v2 = original_edges[1].opposite(order)

                    '''
                    # break edges and remove vertex from origin route
                    print(improved_origin_route.remove_vertex(order))
                    print('\n')
                    improved_origin_route.remove_edge(original_edges[0])
                    improved_origin_route.remove_edge(original_edges[1])
                    # connect the loose vertices in origin route
                    improved_origin_route.add_edge(v1, v2, get_distance_between_vertices(v1.element().coords, v2.element().coords))
                    '''

                    # insert order between two orders in a different route
                    improved_other_route.add_existing_vertex(order)
                    improved_other_route.remove_edge(improved_other_route.get_edge(previous, current))
                    
                    improved_other_route.add_edge(previous, order, get_distance_between_vertices(previous.element().coords, order.element().coords))
                    improved_other_route.add_edge(order, current, get_distance_between_vertices(order.element().coords, current.element().coords))

                    # improved_routes[j] = improved_other_route

                    # find the new position with the lowest cost
                    if best_new_position_route == None or \
                    get_route_distance(improved_other_route) < get_route_distance(best_new_position_route[0]):
                        best_new_position_route = [improved_other_route, j, previous.element().id, current.element().id]
                    '''
                    # make reparations
                    # break edges and remove vertex from origin route
                    improved_origin_route.add_existing_vertex(order)
                    improved_origin_route.remove_edge(improved_origin_route.get_edge(v1, v2))
                    improved_origin_route.add_edge(v1, order, 0)
                    improved_origin_route.add_edge(order, v2, 0)
                    
                    ####
                    original_edges = improved_other_route.get_edges(order)

                    v1 = original_edges[0].opposite(order)
                    v2 = original_edges[1].opposite(order)

                    # break edges and remove vertex from origin route
                    print(improved_other_route.remove_vertex(order))
                    print('\n')
                    improved_other_route.remove_edge(original_edges[0])
                    improved_other_route.remove_edge(original_edges[1])
                    # connect the loose vertices in origin route
                    improved_other_route.add_edge(v1, v2, get_distance_between_vertices(v1.element().coords, v2.element().coords))
                    '''
        # move the order to the cheapest other route 
        # test to see if the overall route cost has improved
            
            original_cost = get_overall_distance(routes)
            print(routes.index(other_routes[best_new_position_route[1]]))
            print(cost_of_break(order, origin_route))
            print('New route distance')
            print(get_route_distance(best_new_position_route[0]))
            t.append([cost_of_break(order, origin_route) + get_route_distance(best_new_position_route[0]), get_route_distance(other_routes[best_new_position_route[1]]), cost_of_break(order, origin_route)])
            g.append([cost_of_break(order, origin_route) + get_route_distance(best_new_position_route[0]), get_route_distance(routes[routes.index(other_routes[best_new_position_route[1]])])])
            if (cost_of_break(order, origin_route) + get_route_distance(best_new_position_route[0])) \
                < (get_route_distance(routes[routes.index(other_routes[best_new_position_route[1]])])):

                print('success')
                time.sleep(1)
                local_search_actioned = True

                original_edges = routes[i].get_edges(order)
                print(original_edges)
                time.sleep(1)
                v1 = original_edges[0].opposite(order)
                v2 = original_edges[1].opposite(order)

                routes[i].remove_edge(original_edges[0])
                routes[i].remove_edge(original_edges[1])
                # connect the loose vertices in origin route
                routes[i].add_edge(v1, v2, get_distance_between_vertices(v1.element().coords, v2.element().coords))


                #-------------------------------
                r_index = routes.index(other_routes[best_new_position_route[1]])
                print('index %i' % r_index)
                # routes[r_index] = best_new_position_route[0]

                previous = routes[r_index].get_vertex_by_id(best_new_position_route[2])
                current = routes[r_index].get_vertex_by_id(best_new_position_route[3])
                
                print(routes[r_index].add_existing_vertex(order))
                print('get_edge')
                print(routes[r_index].get_edge(previous, current))
                print(routes[r_index].remove_edge(routes[r_index].get_edge(previous, current)))
                print('HERE')
                print(routes[r_index].add_edge(previous, order, get_distance_between_vertices(previous.element().coords, order.element().coords)))
                print(routes[r_index].add_edge(order, current, get_distance_between_vertices(order.element().coords, current.element().coords)))
                #routes[r_index].remove_edge(improved_other_route.get_edge(previous, current))
                    
                #routes[r_index].add_edge(previous, order, get_distance_between_vertices(previous.element().coords, order.element().coords))
                #routes[r_index].add_edge(order, current, get_distance_between_vertices(order.element().coords, current.element().coords))

                #improved_routes[j] = improved_other_route



            else:
                print('fail')
                print(get_route_distance(origin_route) + cost_of_break(order, origin_route) + \
                get_route_distance(best_new_position_route[0]))
                print((get_route_distance(origin_route) + \
                get_route_distance(routes[routes.index(other_routes[best_new_position_route[1]])])))
                print(cost_of_break(order, origin_route))
            



        #routes = best_routes_for_order
    print(t)
    print('\n')
    print(g)
    # routes[i].erase()
    return (routes, local_search_actioned)

def grasp(orders):

    L_points, L = GrahamScan(orders)
    L_comp = list(set(orders) - set(L))
    #plot_convex_hull(orders, L_points, '.r')
    S, L, L_comp, routes = route_initialization(L, L_comp)
    unscheduled_orders = L + L_comp

    #plot_convex_hull(S, L_points, 'bo')
    #plot_convex_hull(routes.vertices(), L_points, 'bo')


    routes = main_routing(unscheduled_orders, S, routes)
    improved_routes = two_opt_route_improve(routes)
    plot_routes(orders, improved_routes, "2-opt improved routes")
    with open("distances.txt", "a") as f:
        f.write("%f\n" % get_overall_distance(improved_routes))
    
    print('\n\n')
    print(get_overall_distance(improved_routes))
    print('\n\n')

    local_search_routes, local_search_actioned = local_search2(improved_routes.copy())

    with open("distances.txt", "a") as f:
        f.write("%f\n" % get_overall_distance(local_search_routes))
    

    print('\n\n')
    print(get_overall_distance(local_search_routes))
    print('\n\n')
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
    
    # local_search_routes = local_search2(improved_routes.copy())
    print(local_search_routes == improved_routes)
    # plot_routes(orders, routes, "initial greedy routes")
    
    if local_search_actioned:
        print('YES')
        # final_routes = two_opt_route_improve(local_search_routes)
        # plot_routes(orders, final_routes, "local-search improved routes")
        plot_routes(orders, local_search_routes, "local-search improved routes")
        print('\n\n')
        print(get_overall_distance(final_routes))
        print('\n\n')
        with open("distances.txt", "a") as f:
            f.write("%f" % get_overall_distance(final_routes))

    with open("distances.txt", "a") as f:
            f.write("\n\n######\n\n")
    

    

def plot_routes(orders, routes, title="untitled"):

    plt.figure()
    plt.suptitle(title, fontsize=20)
    P = list()
    for order in orders:
        P.append(order.coords)
    plt.scatter(*zip(*P))
    P = np.array(P)

    # plt.plot(P[:,0],P[:,1], 'b-', picker=5)
    #plt.plot([P[-1,0],P[0,0]],[P[-1,1],P[0,1]], 'b-', picker=5)
    #plt.plot(P[:,0],P[:,1], 'b')
    plt.plot(DEPOT_COORDS[0], DEPOT_COORDS[1], '*g')
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
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
            plt.plot(x_values, y_values, colors[i])
        i += 1


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

