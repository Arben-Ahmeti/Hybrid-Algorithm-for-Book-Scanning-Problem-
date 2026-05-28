class Book:
    __slots__ = ('id', 'score')

    def __init__(self, id, score):
        self.id = id
        self.score = score

    def __repr__(self):
        return f"Book({self.id}, {self.score})"
