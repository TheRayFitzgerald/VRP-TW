if origin_route.get_edges(v1)[0] == None:
                    print('v1 remove none')
                    print(origin_route.get_edges(v1))
                    #print(routes[i].remove_edge(origin_route.get_edges(v1)[0]))
                    print(origin_route.get_edges(v1))
                    time.sleep(5)
                if origin_route.get_edges(v2)[0] == None:
                    print('v2 remove none')
                    print(origin_route.get_edges(v2))
                    print(routes[i].remove_edge(origin_route.get_edges(v2)[0]))
                    print(origin_route.get_edges(v2))
                    time.sleep(5)


def local_search_tw_shuffle_iterator(routes, n_rounds):

    open('num_vertices.txt', 'w').close()
    num_vertices = 0
    for route in routes:
        num_vertices += route.num_vertices() - 1
    with open("num_vertices.txt", "a") as f:
        f.write("Initial: %i\n" % num_vertices)

    timeout = time.time() + 10
    best_distance = get_overall_distance(routes)
    for i in range(n_rounds):
        while get_overall_distance(routes) >= best_distance:
            routes = local_search_tw(routes)[0]
            num_vertices = 0
            for route in routes:
                num_vertices += route.num_vertices() - 1
            if num_vertices < 40:
                print('Local Search')
                time.sleep(20)

            routes, failed_vertices = tw_shuffle(routes)
            num_vertices = 0
            for route in routes:
                num_vertices += route.num_vertices() - 1
            if num_vertices < 40:
                print('TW Shuffle')
                time.sleep(20)

            
        num_vertices = 0
        for route in routes:
            num_vertices += route.num_vertices() - 1
        with open("num_vertices.txt", "a") as f:
            f.write("%i\n" % num_vertices)

        best_distance = get_overall_distance(routes)
        plot_routes(routes, "Round %i" % (i+1))

    return routes

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

def tw_shuffle_new(routes):

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
            
            # continue until all shuffled vertices have been assigned.
            while shuffled_vertices:

                # randomly pick a vertex to insert
                vertex = random.choice(shuffled_vertices)
                # create a list to record all feasible routes for this vertex.
                feasible_routes = list()
                # iterate through all edges to find the best position for this vertex
                for edge in improved_route.edges():
                    
                    start = edge.start()
                    end = edge.end()
                    improved_route.add_vertex_between_vertices(vertex, start, end)

                    print('$$$$ ORIGINAL $$$$\n')
                    print(route)
                    #print('Distance: %f' % get_route_distance(improved_other_route))
                
                    print('\n----- Origin -----')
                    print('Order: %i' % (vertex.element().id))
                    print('------------------')

                    print('\n----- Destination -----')
                    print('Order: %i <-> %i' % (start.element().id, end.element().id))
                    print('-----------------------\n')
                    
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
                    shuffled_vertices.remove(vertex)

                # vertex inserted into its best position - continue

                # no routes found for this vertex
                except Exception as e:
                    print('FEASIBLE ROUTES')
                    print(feasible_routes)
                    
                    print(e)
                    failed_vertices_for_route.append(vertex)
                    #time.sleep(5)
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
                    '''
                    print('$$$$ START $$$$\n')
                    print(improved_other_route)
                    print('Distance: %f' % get_route_distance(improved_other_route))
                
                    print('\n----- Origin -----')
                    print('Route: %s, Order: %i' % (colors[i], order.element().id))
                    print('------------------')

                    print('\n----- Destination -----')
                    print('Route: %s, Order: %i <-> %i' % (colors[routes.index(other_routes[j]) % 7], start.element().id, end.element().id))
                    print('-----------------------\n')
                    '''
                    improved_routes = routes.copy()
                    improved_origin_route = copy.deepcopy(improved_routes[i])

                    if origin_route.degree(order) > 2:
                        # time.sleep(1)
                        print('TOO MANY EDGES')
                    '''
                    print('$$$$ Origin Route $$$$\n')
                    print(origin_route)
                    '''
                    improved_other_route.add_vertex_between_vertices(order, start, end)
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
