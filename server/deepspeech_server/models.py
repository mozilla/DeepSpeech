class Response:
    def __init__(self, text, time):
        self.text = text
        self.time = time


class Error:
    def __init__(self, message):
        self.message = message
