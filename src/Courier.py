
class Courier:

    def __init__(self, name):

        self._name = name
        self._van = ''

    def __str__(self):
        return ('Courier: ' + self._name + '\nVan: ' + str(self._van))

    @property
    def van(self):
        return self._van

    @van.setter
    def van(self, van):
        self._van = van
