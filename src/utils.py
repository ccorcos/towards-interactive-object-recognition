class ex(Exception):

    def __init__(self, msg):
        Exception.__init__(self, msg)


def wait(msg=""):
    print msg
    _ = raw_input("Press <enter> to continue...")
