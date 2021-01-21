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