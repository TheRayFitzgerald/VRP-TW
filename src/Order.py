import time, datetime
import random
from math import sqrt

DEPOT_COORDS = (150, 150)
# aribtrary speed in km/h
SPEED = 3

class Order:

    def __init__(self, id, user, coords, scheduled_time, size):

        self._id = id
        self._uid = random.randint(0,1000)
        self._user = user
        self._coords = coords
        self._distance = sqrt((DEPOT_COORDS[0] - self._coords[0])**2 + (DEPOT_COORDS[1] - self._coords[1])**2)
        self._placed_time = time.time()
        self._scheduled_time = scheduled_time
        self._time_to_delivery = random.randrange(30)
        self._slack = scheduled_time - (datetime.timedelta(hours=9) + datetime.timedelta(minutes=round(self._distance / SPEED)))
        self._size = size
        self._seed = False

    def __str__(self):

        return str(self.id)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def uid(self):
        return self._uid

    @uid.setter
    def uid(self, uid):
        self._uid = uid

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

    @scheduled_time.setter
    def scheduled_time(self, scheduled_time):
        self._scheduled_time = scheduled_time

    @property
    def size(self):
        return self._size

    @property
    def distance(self):
        return self._distance

    @property
    def time_to_delivery(self):
        return self._time_to_delivery

    @property
    def slack(self):
        return self._slack

    @slack.setter
    def slack(self, slack):
        self._slack = slack

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed






