"""
Ray Fitzgerald
117381503
"""


class Element:
    """ A key, value and index.
     Where key is the priority(distance),
     value is the vertex itself,
     and index is it's location in the APQ"""

    def __init__(self, k, v, i):
        self._key = k
        self._value = v
        self._index = i

    def __str__(self):
        description = ''
        description += '\nKey: %s' % str(self._key)
        description += ' Value: %s ' % str(self._value)
        description += ' Index: %s' % str(self._index)
        return description

    def __eq__(self, other):
        return self._key == other._key

    def __lt__(self, other):
        return self._key < other._key

    def _wipe(self):
        self._key = None
        self._value = None
        self._index = None


class APQ:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def __str__(self):
        description = ''
        for i in range(1, self.currentSize + 1):
            description += "\n %s" % self.heapList[i]

        return description

    def bubbleUp(self, i):

        # checks if a parent exists
        while i // 2 > 0:

            # if key is less than that of its parents. swap
            if self.heapList[i]._key < self.heapList[i // 2]._key:
                tmp = self.heapList[i // 2]
                self.heapList[i // 2]._index, self.heapList[i]._index = \
                    self.heapList[i]._index, self.heapList[i // 2]._index
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
                i = i // 2
            else:
                return i
        return i

              # change to parent index and check again

    # add an item to the APQ
    def add(self, k, v):
        # create an object from the Element Class
        item = Element(k, v, self.currentSize)
        self.heapList.append(item)
        self.currentSize = self.currentSize + 1

        return self.heapList[self.bubbleUp(self.currentSize)]

    def min(self):

        return self.heapList[1]

    def bubbleDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)  # index of min child
            if self.heapList[i]._key > self.heapList[mc]._key:
                tmp = self.heapList[i]
                self.heapList[i]._index, self.heapList[mc]._index = self.heapList[mc]._index, self.heapList[i]._index
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    def minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2]._key < self.heapList[i * 2 + 1]._key:

                 # if i * 2 + 2 < self.currentSize and self.heapList[i * 2 + 2]._key < self.heapList[i * 2 + 1]._key:
                   #  return i * 2 + 2
                return i * 2
            else:
                return i * 2 + 1

    def remove_min(self):
        rem = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.heapList[1]._index = 0
        self.currentSize = self.currentSize - 1
        self.heapList.pop()  # remove min val from men
        self.bubbleDown(1)
        return rem

    def update_key(self, element, newkey):

        oldkey = self.get_key(element)
        self.heapList[element._index + 1]._key = newkey

        if newkey < oldkey:
            self.bubbleUp(element._index + 1)

        else:
            self.bubbleDown(element._index + 1)

    def remove(self, element):

        oldkey = self.get_key(element)
        index = element._index + 1
        last = self.heapList[self.currentSize]
        newkey = last._key
        self.heapList[index], last = last, self.heapList[index]
        self.heapList[index]._index = last._index  # update to correct index
        self.currentSize -= 1
        self.heapList.pop()

        if newkey < oldkey:
            self.bubbleUp(index)
        else:
            self.bubbleDown(index)

    def get_key(self, element):

        return self.heapList[element._index + 1]._key

    def is_empty(self):

        if self.currentSize == 0:
            return True
        return False

    def length(self):
        return self.currentSize
