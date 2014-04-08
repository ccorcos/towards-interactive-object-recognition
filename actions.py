from pylab import *

# Karol's C++ program will gather data and compute errors. This must only
# deal with probabilities, etc.

# or a function to set these
objects = ['book1', 'book2', 'book3']
poses = ['up-forward', 'up-backward', 'down-forward', 'down-backward']


# 		1
# p1 =	0
# 		0
# 		0

# 		0
# p2 = 	1
# 		0
# 		0

# 		0
# p3 = 	0
# 		1
# 		0

# 		0
# p4 = 	0
# 		0
# 		1


def pose2vec(name):
    vec = zeros(len(poses))
    vec[poses.index(name)] = 1
    return vec


def vec2pose(vec):
    return poses[vec.index(1)]

actions = ['stay', 'flip', 'rotate', 'flipRotate']

# suppose the next pose is computed by pi' = ai*pi
# 		1 0 0 0
# a1 = 	0 1 0 0
# 		0 0 1 0
# 		0 0 0 1

# 		0 0 1 0
# a2 = 	0 0 0 1
# 		1 0 0 0
# 		0 1 0 0

# 		0 1 0 0
# a3 = 	1 0 0 0
# 		0 0 0 1
# 		0 0 1 0

# 		0 0 0 1
# a4 = 	0 0 1 0
# 		0 1 0 0
# 		1 0 0 0

stay = eye(4)
flip = zeros([4, 4])
flip[0, 2] = flip[1, 3] = flip[2, 0] = flip[3, 1] = 1
rotate = zeros([4, 4])
rotate[0, 1] = rotate[1, 0] = rotate[2, 3] = rotate[3, 2] = 1
flipRotate = eye(4)[::-1]


def action2mat(name):
    if name == 'stay':
        return stay
    elif name == 'flip':
        return flip
    elif name == 'rotate':
        return rotate
    elif name == 'flipRotate':
        return flipRotate
    else:
        raise "invalid action name, " + name


def mat2action(mat):
    if array_equal(mat, stay):
        return'stay':
    elif array_equal(mat, flip):
        return'flip':
        elif array_equal(mat, rotate):
        return'rotate':
    elif array_equal(mat, flipRotate):
        return'flipRotate':
    else:
        raise "invalid action matrix, " + str(mat)


def emptyArray(shape):
    # initialize an empty array with a defined shape
    a = None
    for i in shape[::-1]:
        a = [a] * i
    return array(a)


def wrap(func, args):
    # make a wrapper to pass list of args as args
    func(*args)


class ArrayCache:

    def __init__(self, shape, fcn):
        self.cache = emptyArray(shape)
        self.fcn = fcn

    def __str__(self):
        return str(self.cache.tolist)

    def __getitem__(self, index):
        items = self.cache.__getitem__(index)
        for i in items:
            if i == None:

# For now, we will assume all features are SURF features and modeled as
# Gaussian


class Gaussian:

    def __init__(self, values):
    # values = array[sample][value]
        v = array(values)
        self.m = mean(values, 0)
        self.c = cov(valuee.T)

    def prob(self, value):
        v = array(value)
        d = len(self.m)
        return pow(2 * pi, -d / 2.0) / sqrt(det(self.c)) * \
            exp(-0.5 * (v - self.m) * pinv(self.c) * (v - self.m).T)

    def expectedValue(self):
        return self.mu


def train(errors):
    # errors = array[object][pose][feature][sample]
    # This function learns a distribution, p(f|o,p)

    global nObjects, nPoses, nFeatures, nTrainingSamples, nActions, nObservations
    nObjects, nPoses, nFeatures, nTrainingSamples = errors.shape
    nActions = nObservations = nPoses

    initCaches()

    for n in range(N):
        for i in range(I):
            for m in range(M):
                dfgop[n, i, m] = Gaussian(errors[n, i, m, :])

    # cache_dFgop initialize
    # cache_dF initialize


def initCaches():
        # initalize some caches and constants
    global cache_dfgop  # learned distribution
    cache_dfgop = emptyArray([nObjects, nPoses, nFeatures])

    global cache_dFgop  # learned joint distribution
    cache_dFgop = emptyArray([nObjects, nPoses])

    global cache_pFgop
    cache_pFgop = emptyArray([nObservations, nObjects, nPoses])

    global cache_F  # observed features
    cache_F = emptyArray([nObservations, nFeatures])

    global cache_prob_evidence
    cache_prob_evidence = emptyArray([nObservations])

    global cache_dist_evidence
    cache_dist_evidence = emptyArray([nObservations])

    global cache_pFgop
    cache_pFgop = emptyArray([nObservations, nObjects, nPoses])

    global pop
    pop = 1.0 / float(nObjects * nPoses)

    global cache_posterior_prob_op
    cache_posterior_prob_op = emptyArray([nObservations, nObjects, nPoses])

    global cache_posterior_dist_op
    cache_posterior_dist_op = emptyArray([nObservations, nObjects, nPoses])

    global cache_posterior_prob_o
    cache_posterior_prob_o = emptyArray([nObservations, nObjects])

    global cache_posterior_dist_o
    cache_posterior_dist_o = emptyArray([nObservations, nObjects])


def observe(errors):
    # errors = array[feature]
    # solve for the posterior

    # if the first observation
    if not cache_pFgop[0, 0, 0]:
        for obj in range(nObjects):
            for pose in range(nPoses):
                cache_posterior_prob_op[0, obj, pose] = pop/
    else:
        pass


def observeFirst(errors):
    # errors = array[feature]
    # solve for the posterior
