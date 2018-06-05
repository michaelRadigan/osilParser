
class ObjectiveFunction(object):

    def __init__(self, weight, name, maxOrmin, linear=None, quadratic=None):
        self.weight = weight
        self.name = name
        self.maxOrMin = maxOrmin
        if quadratic is not None:
            quadratic = quadratic
        self.quadratic = quadratic
        if linear is not None:
            self.linear = linear

