
class DataIterator:
    """
    Custom iterator to get n chunks of two arrays (features
    and labels) at a time
    """
    def __init__(self, data1, data2, n):
        assert data1.shape[0] == data1.shape[0]
        self.data1 = data1
        self.data2 = data2
        self.n = n
        self.first_index, self.last_index = -1, -1
        if len(data1) > 0:
            self.first_index = 0
        if len(data1) > n - 1:
            self.last_index = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.last_index == -1:
            chunk1 = self.data1[self.first_index:]
            chunk2 = self.data2[self.first_index:]
        else:
            chunk1 = self.data1[self.first_index:self.last_index]
            chunk2 = self.data2[self.first_index:self.last_index]
        if self.first_index == -1:
            raise StopIteration
        self.first_index = self.last_index
        if self.last_index + self.n < len(self.data1):
            self.last_index += self.n
        else:
            self.last_index = -1
        return chunk1, chunk2
