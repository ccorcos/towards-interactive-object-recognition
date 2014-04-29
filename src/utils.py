class ex(Exception):

    def __init__(self, msg):
        Exception.__init__(self, msg)


def wait(msg=""):
    if msg != "":
        print ""
        print msg
    else:
        print ""
    _ = raw_input("Press <enter> to continue...")
    print ""


import pprint


def pr(level, *args, **kwargs):
    '''
    @summary: helper function for printing to terminal
    @param level: defines how to tabulate the print statement
    @param *args: numbers, strings, anything
    @result: nothing
    '''

    # if "pretty" in kwargs:

    string = " ".join([pprint.pformat(arg) for arg in args])
    string = string.replace("'", "").replace('"', '')
    if level == 0:
        sym = "="
        w = 78
        string = " " + string + " "
        n = len(string)
        d = w - n
        l = ''
        r = ''
        if d > 0:
            if d % 2 == 0:
                l = sym * int(d / 2.)
                r = l
            else:
                l = sym * int(d / 2.)
                r = sym * int(d / 2. + 1)
            print ''
            print sym * w
            print l + string + r
            print sym * w
            print ''
        else:
            print ''
            print sym * w
            print string
            print sym * w
            print ''
    elif level == 1:
        print " - " + string
    elif level == 2:
        print "   * " + string
    elif level == 3:
        print "     = " + string
    elif level == 4:
        print "       > " + string
    elif level == 4:
        print "         + " + string
    elif level == 100:
        lines = string.split("\n")
        for line in lines:
            print "      " + line
    else:
        print "? " + string
