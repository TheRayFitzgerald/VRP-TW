import time


class Order:

    def __init__(self, id, user, address, placed_time, scheduled_time, size):

        self._id = id
        self._user = user
        self._address = address
        self._placed_time = time.time()
        self._scheduled_time = scheduled_time
        self._size = size

    @property
    def id(self):
        return self._id

    @property
    def user(self):
        return self._user

    @property
    def get_address(self):
        return self._address

    @property
    def placed_time(self):
        return self._placed_time

    @property
    def scheduled_time(self):
        return self._scheduled_time

    @property
    def size(self):
        return self._size
