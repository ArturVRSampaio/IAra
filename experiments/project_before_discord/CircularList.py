
class CircularList:
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []

    def add(self, item):
        if len(self.data) >= self.max_size:
            self.data.pop(0)
        self.data.append(item)

    def mean(self):
        if not self.data:
            return 0
        return sum(self.data) / len(self.data)

    def __len__(self):
        return len(self.data)
