
class User:

    def __init__(self, id, name, address):

        self._id = id
        self._name = name
        self._address = adddress
        self._orders = orders

    @property
    def orders(self):
        return self._orders

    @orders.setter
    def orders(self, order):
        self._orders.append(order)
