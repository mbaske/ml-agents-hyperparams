# # Unity ML Agents
#
# Added to enable automated hyperparameter tuning.
# M.Baske, July 2018


class TrainerEvent(object):

    def __init__(self, default_handler):
        self.handlers = [default_handler]
        self.trainer_num = 0

    def __exit__(self):
        self.handlers = []

    def add(self, handler):
        self.handlers.append(handler)
        return self

    def remove(self, handler):
        self.handlers.remove(handler)
        return self

    def dispatch(self, data=None):
        for handler in self.handlers:
            handler(data)
