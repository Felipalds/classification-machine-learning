
class Analysis:

    def __init__(self, data):
        self.data = data

    def intro(self):
        print("Welcome to the analysis module!")
        print(self.data.head())
