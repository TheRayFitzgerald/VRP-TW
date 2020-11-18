import time
import random


class Order:

    def __init__(self, id, user, coords, placed_time, scheduled_time, size):

        self._id = id
        self._user = user
        self._coords = coords
        self._distance = random.randrange(30)
        self._placed_time = time.time()
        self._scheduled_time = scheduled_time
        self._time_to_delivery = random.randrange(30)
        self._size = size

    def __str__(self):

        return str(self.id)

    @property
    def id(self):
        return self._id

    @property
    def user(self):
        return self._user

    @property
    def coords(self):
        return self._coords

    @property
    def placed_time(self):
        return self._placed_time

    @property
    def scheduled_time(self):
        return self._scheduled_time

    @property
    def size(self):
        return self._size

    @property
    def distance(self):
        return self._distance

    @property
    def time_to_delivery(self):
        return self._time_to_delivery



