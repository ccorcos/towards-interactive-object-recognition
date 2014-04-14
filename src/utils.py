class ex(Exception):

    def __init__(self, msg):
        Exception.__init__(self, msg)


class memorize(dict):

    """cache all the function calls"""

    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result


def wait(msg=""):
    print msg
    _ = raw_input("Press <enter> to continue...")
