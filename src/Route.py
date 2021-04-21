from random import randint
from math import sqrt
import time, datetime, sys
from read_config import read_config
from Order import Order

START_TIME = datetime.timedelta(hours=9)
DEPOT_COORDS = (150, 150)

config_vars = read_config()
for key,val in config_vars.items():
    exec(key + '=val')

class Edge:
    """ An edge in a graph.

        Implemented with an order, so can be used for directed or undirected
        graphs. Methods are provided for both. It is the job of the Route class
        to handle them as directed or undirected.
    """

    def __init__(self, o1, o2):
        """ Create an edge between vertices v and w, with a data element.

        Element can be an arbitrarily complex structure.

        Args:
            element - the data or label to be associated with the edge.
        """
        self._orders = (o1, o2)
        self._distance = sqrt((o1.coords[0] - o2.coords[0])**2 + (o1.coords[1] - o2.coords[1])**2)

    def __str__(self):
        """ Return a string representation of this edge. """
        return ('(' + str(self._orders[0]) + '--'
                + str(self._orders[1]) + ' : '
                + str(self._distance) + ')')

    def __repr__(self):
        return str(self)

    def orders(self):
        """ Return an ordered pair of the vertices of this edge. """
        return self._orders

    def start(self):
        """ Return the first vertex in the ordered pair. """
        return self._orders[0]

    def end(self):
        """ Return the second vertex in the ordered pair. """
        return self._orders[1]

    def opposite(self, o):
        """ Return the opposite vertex to v in this edge.

        Args:
            v - a vertex object
        """
        if self._orders[0] == o:
            return self._orders[1]
        elif self._orders[1] == o:
            return self._orders[0]
        else:
            return None

    @property
    def distance(self):
        """ Return the data element for this edge. """
        return self._distance


class Route:
    """ Represent a simple graph.

    This version maintains only undirected graphs, and assumes no
    self loops.
    """

    # Implement as a Python dictionary
    #  - the keys are the vertices
    #  - the values are the sets of edges for the corresponding vertex.
    #    Each edge set is also maintained as a dictionary,
    #    with the opposite vertex as the key and the edge object as the value.

    def __init__(self, order):
        """ Create an initial empty graph. """
        # depot_to_first_order_edge
        depot = Order(0, 'Depot', DEPOT_COORDS, START_TIME, 0)
        e = Edge(depot, order)
        self._structure = dict()
        self._structure[depot] = {order: e}
        self._structure[order] = {depot: e}
            

    def __str__(self):
        """ Return a string representation of the graph. """
        hstr = ('|Orders| = ' + str(self.num_orders())
                + '; |Edges| = ' + str(self.num_edges()))
        vstr = '\nOrder: '
        for v in self._structure:
            vstr += str(v) + '-'
        edges = self.edges()
        estr = '\nEdges: '
        for e in edges:
            estr += str(e) + ' '
        return hstr + vstr + estr

    def __repr__(self):
        return '\n' + str(self) + '\n'

    # -----------------------------------------------------------------------#
    @property
    def id(self):
        return self._id
    # ADT methods to query the graph

    def num_orders(self):
        """ Return the number of vertices in the graph. """
        return len(self._structure)

    def num_edges(self):
        """ Return the number of edges in the graph. """
        num = 0
        for v in self._structure:
            num += len(self._structure[v])  # the dict of edges for v
        return num // 2  # divide by 2, since each edege appears in the
        # vertex list for both of its vertices

    def orders(self):
        """ Return a list of all vertices in the graph. """
        return [key for key in self._structure]

    def get_order(self, o):

        for order in self.orders():
            if order == o:
                return order

    def get_order_by_uid(self, uid):
        for o in self._structure:
            if o.uid == uid:
                return o
        return None

    def get_distance_between_orders(self, o1, o2):

        return sqrt((o1.coords[0] - o2.coords[0])**2 + (o1.coords[1] - o2.coords[1])**2)

    def distance_to_order(self, order):

        total_distance = 0
        edges = self.edges_in_order()

        for edge in edges:

            start = edge.start()
            end = edge.end()

            total_distance += self.get_distance_between_orders(start, end)

            if start == order or end == order:
                break

        return total_distance

    def order_is_reachable(self, order, edges):
        total_time = 0
        
        for edge in edges:
            #print(edge)
            start = edge.start()
            end = edge.end()

            total_time += self.get_distance_between_orders(start, end) / SPEED

            if start == order or end == order:
                break

        total_time = datetime.timedelta(minutes=total_time)
        
        # store if it is reachable or not
        reachable = START_TIME + total_time < order.scheduled_time

        return reachable

    def edges(self):
        """ Return a list of all edges in the graph. """
        edgelist = []
        for v in self._structure:
            for w in self._structure[v]:
                # to avoid duplicates, only return if v is the first vertex
                if self._structure[v][w].start() == v:
                    edgelist.append(self._structure[v][w])
        return edgelist

    def edges_in_order(self):
        """ Return a list of all edges *in order* starting and finishing at the depot in the graph. """
        
        edges = self.edges_in_order_undirected()
    
        for order in self.orders()[1:]:
            if not self.order_is_reachable(order, edges):
                edges.reverse()
                for order in self.orders()[1:]:
                    if not self.order_is_reachable(order, edges):
                        sys.exit('INFEASIBLE ROUTE')
                edges.reverse()
                break
        
        return edges

    def is_feasible(self):

        edges = self.edges_in_order_undirected()
    
        for order in self.orders()[1:]:
            if not self.order_is_reachable(order, edges):
                edges.reverse()
                for order in self.orders()[1:]:
                    if not self.order_is_reachable(order, edges):
                        return False
                
        return True

    def edges_in_order_undirected(self):
        """ Return a list of all edges *in order* starting and finishing at the depot in the graph. """
        
        # start by getting an edge with depot in it
 
        edges = [self.get_edges(self.orders()[0])[0]]
        # then add the next edge in the correct direction
        for edge in self.get_edges(edges[-1].opposite(self.orders()[0])):
            if edge not in edges:
                edges.append(edge)
        try:
            # iterate over until we get all of the edges
            while len(edges) != self.num_edges():
                for order in edges[-1].orders():
                    for edge in self.get_edges(order):
                        if edge not in edges:
                            edges.append(edge)

        except Exception as e:
            print(e)
            time.sleep(3)
            raise Exception('ex')

        return edges

    def get_edges(self, o):
        """ Return a list of all edges incident on v.

        Args:
            v - a vertex object
        """
        if o in self._structure:
            edgelist = []
            for o1 in self._structure[o]:
                edgelist.append(self._structure[o][o1])
            return edgelist
        return None

    def get_edge(self, v, w):
        """ Return the edge between v and w, or None.

        Args:
            v - a vertex object
            w - a vertex object
        """
        if (self._structure is not None
                and v in self._structure
                and w in self._structure[v]):
            return self._structure[v][w]
        return None

    def degree(self, v):
        """ Return the degree of vertex v.

        Args:
            v - a vertex object
        """
        return len(self._structure[v])

    # ----------------------------------------------------------------------#

    # ADT methods to modify the graph

    def add_order(self, order):

        for o in self._structure:
            if o == order:
                return o
        self._structure[order] = dict()
        return order

    def add_order_between_orders(self, o, o1, o2):

        edgelist = [edge for edge in self.get_edges(self.orders()[0]) if edge.start() == self.orders()[0]]
    
        self.remove_edge(self.get_edge(o1, o2))
        order = self.add_order(o)
        if o2.coords == (150, 150) and len(edgelist) == 0:
            self.add_edge(o2, o)
            self.add_edge(o, o1)
        else:
            self.add_edge(o1, o)
            self.add_edge(o, o2)
        return order

    def remove_order(self, order):

        return self._structure.pop(order, None)

    def remove_order_and_repair(self, order):

        try:
            if self.degree(order) > 1:                
                o1 = self.get_edges(order)[0].opposite(order)
                o2 = self.get_edges(order)[1].opposite(order)

                # remove surroundin edges and the vertex itself
                self.remove_edge(self.get_edges(order)[0])
                self.remove_edge(self.get_edges(order)[0])
                self.remove_order(order)

                # make reparations
                self.add_edge(o1, o2)

            # The order's degree is <= 1. Therefore it is the only order in the route.
            # Therefore, we want to remove the route altogether.
            else:
                self.remove_edge(self.get_edges(order)[0])
                self.remove_order(order)

        except Exception as e:
            print('here')
            print(e)
            time.sleep(5)

    def add_edge(self, o1, o2):
        """ Add and return an edge between two vertices v and w, with  element.

        If either v or w are not vertices in the graph, does not add, and
        returns None.

        If an edge already exists between v and w, this will
        replace the previous edge.

        Args:
            v - a vertex object
            w - a vertex object
            element - a label
        """
        if o1 not in self._structure:
            print('v')
            # time.sleep(10)
            return None
        if o2 not in self._structure:
            print('w')
            # time.sleep(10)
            return None

        e = Edge(o1, o2)
        try:
            self._structure[o1][o2] = e
            self._structure[o2][o1] = e
        except Exception as e:
            print(e)
            print('cee')
        return e

    def remove_edge(self, edge):

        try:
            o1 = edge.orders()[0] 
            o2 = edge.orders()[1]

            try:
                o1_to_o2 = self._structure[o1].pop(o2, None)
                o2_to_o1 = self._structure[o2].pop(o1, None)
            except Exception as e:
                print(e)

            if edge1 and edge2:
                return True

            return None

        except Exception as e:
            #print(e)
            return None

    # method to find the single vertex with only 1 edge(as opposed to 2)
    # method connects this vertex back to the origin/depot vertex.
    def complete_route(self):
        for order in self.orders():
            if self.degree(order) == 1 and order != self.orders()[0]:
                self.add_edge(order, self.orders()[0])

    def erase(self):
        self._structure = dict()

