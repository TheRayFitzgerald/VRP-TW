# Author: Rodolfo Ferro
# Mail: ferro@cimat.mx
# Script: Compute the Convex Hull of a set of points using the Graham Scan

import sys
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

# Function to know if we have a CCW turn
def RightTurn(p1, p2, p3):
	if (p3[1]-p1[1])*(p2[0]-p1[0]) >= (p2[1]-p1[1])*(p3[0]-p1[0]):
		return False
	return True

# Main algorithm:
def GrahamScan(orders):
    obj_coords = list()
    for order in orders:
       obj_coords.append([order, order.coords])

	#P.sort()			# Sort the set of points
    P = sorted(obj_coords, key=itemgetter(1))

    L_upper = [P[0], P[1]]

    for i in range(2,len(P)):
        L_upper.append(P[i])
        while len(L_upper) > 2 and not RightTurn(L_upper[-1][1],L_upper[-2][1],L_upper[-3][1]):
            del L_upper[-2]
    L_lower = [P[-1], P[-2]]	# Initialize the lower part
    # Compute the lower part of the hull
    for i in range(len(P)-3,-1,-1):
        L_lower.append(P[i])
        while len(L_lower) > 2 and not RightTurn(L_lower[-1][1],L_lower[-2][1],L_lower[-3][1]):
            del L_lower[-2]
    del L_lower[0]
    del L_lower[-1]
    L = L_upper + L_lower		# Build the full hull
    return_L = list()
    return_orders = list()
    for item in L:
        return_L.append(item[1])
        return_orders.append(item[0])

    return (np.array(return_L), return_orders)

def main():
	try:
		N = int(sys.argv[1])
	except:
		N = int(input("Introduce N: "))

	# By default we build a random set of N points with coordinates in [0,300)x[0,300):
	P = [(np.random.randint(0,300),np.random.randint(0,300)) for i in range(N)]
	L = GrahamScan(P)
	P = np.array(P)

	# Plot the computed Convex Hull:
	plt.figure()
	plt.plot(L[:,0][1],L[:,1][1], 'b-', picker=5)
	plt.plot([L[-1,0][1],L[0,0][1]],[L[-1,1][1],L[0,1][1]], 'b-', picker=5)
	plt.plot(P[:,0][1],P[:,1][1],".r")
	plt.axis('off')
	plt.show()

if __name__ == '__main__':
	main()