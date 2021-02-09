from Graph import Graph
from Order import Order
import datetime

#a = datetime.timedelta(seconds=20)
#b = datetime.timedelta(seconds=5)



a = Order(2, 'ray', (3, 4), datetime.timedelta(seconds=20), 3)
b = Order(2, 'ray', (3, 4), datetime.timedelta(seconds=5), 3)

c = a.scheduled_time - b.scheduled_time
print(c.seconds)

'''
graph = Graph('a')
a = graph.add_vertex('a')
b = graph.add_vertex('b')
c = graph.add_vertex('c')
d = graph.add_vertex('d')
e1 = graph.add_edge(a, b, 1)
e2 = graph.add_edge(a, c, 1)
e3 = graph.add_edge(b, c, 1)
#graph.add_edge(b, d, 1)

print(graph)

print(graph.get_edges(b))

graph.remove_edge(e1)
print(graph.get_edges(b))

print('#')
print(graph.get_edges(a))

graph.remove_edge(e3)
print(graph.get_edges(b))

for j in range(3,4):
	print(j)
'''

