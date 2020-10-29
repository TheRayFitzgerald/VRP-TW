
class Van:

    def __init__(self, capacity):
        self._capacity = capacity
        self._orders = list()

    def __str__(self):
        return str('Capacity ' + str(self._capacity))

    @property
    def capacity(self):
        return self._capacity

    @property
    def orders(self):
        return self._orders

    @orders.setter
    def orders(self, order):
        self._orders.append(order)


if __name__ == '__main__':
    van1 = Van(3)
    print(van1)