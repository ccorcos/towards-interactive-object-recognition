
#
# IMPORT
#

# File format is:
# objectname, posename
# feature1error1, feature1error2, feature1error3,...
# feature2error1, feature2error2, feature2error3,...
# ...
# featureNerror1, featureNerror2, featureNerror3,...
# objectname, posename
# ...
#

from pylab import *

from utils import *
import pprint

# trainingData = {"objectname":{"posename":[feature][training sample]}}
# errors = [objects][poses][features][training samples]
# N objects
# I poses
# J actions
# M features
# R training samples
# J number of observations
# K number of object-poses


objects = []
poses = []
trainingData = {}
errors = []
N = I = J = K = M = R = 0

# FIX: Hard coded actions
actions = ['stay', 'rotate', 'flip', 'flip-rotate']

# first observation at t = 1 = len(observationHistory)-1
# t = 0 represents the prior
observationHistory = [[]]  # no observation or action at t=0
actionHistory = [[]]  # no observation or action at t=0


def whichObservationIdx():
    return len(observationHistory) - 1


def importData():
    """Imports SIFT feature errors to training data. Also builds the
    pgm struture with objects and poses arrays"""
    global objects, poses, trainingData
    print "importing sift feature data"
    f = open("MODEL_SIFT_STANDARD.txt", 'r')
    string = f.read()
    string = string.replace(' ', '').replace(',\n', '\n')
    lines = string.split('\n')[:-1]
    elements = [x.split(',') for x in lines]

    o = ''
    p = ''
    for line in elements:
        if len(line) == 2:
            o = line[0]
            p = line[1]
            # FIX: Hard-coded pose names
            if p == '0':
                p = 'down'
            elif p == '1':
                p = 'down-spine'
            elif p == '2':
                p = 'up-spine'
            elif p == '3':
                p = 'up'
            else:
                raise ex('IMPORT ERROR: unknown pose')
            if o not in objects:
                objects.append(o)
            if p not in poses:
                poses.append(p)
            if o not in trainingData:
                trainingData[o] = {}
            if p not in trainingData[o]:
                trainingData[o][p] = []
        else:
            trainingData[o][p].append([float(i) for i in line])

    pprint.pprint(objects)
    # ['first_home_book',
    #  'first_home_sticker_book',
    #  'math_principles_book',
    #  'math_principles_sticker_book']

    pprint.pprint(poses)
    # ['down', 'down-spine', 'up-spine', 'up']

    # plot
    if False:
        print "plotting training for one object-pose"

        objectPose = np.array(trainingData['first_home_book']['down'])
        cm = get_cmap('gist_rainbow')

        x = np.linspace(0, 500, 1000)
        for i in range(len(objectPose)):
            color = cm(1. * i / len(objectPose))
            t = objectPose[i]
            plt.plot(x, normpdf(x, mean(t), sqrt(var(t))))
        plt.show()

    global N, I, J, K, M, R, errors
    N = len(objects)
    I = len(poses)
    J = len(actions)

    errors = []
    for n in range(N):
        obj = objects[n]
        errors.append([])
        for i in range(I):
            pose = poses[i]
            errors[n].append(trainingData[obj][pose])
    errors = array(errors)
    _, _, M, R = errors.shape

    K = N * I


@memorize
def nextPose(pose, action):
    """Returns the pose after an action is applied to a pose"""
    if pose not in poses:
        raise ex("ERROR: unrecognized pose")
    if action not in actions:
        raise ex("ERROR: unrecognized action")
    if action == 'stay':
        return pose
    elif action == "rotate":
        if pose == 'down':
            return 'down-spine'
        elif pose == 'down-spine':
            return 'down'
        elif pose == 'up':
            return 'up-spine'
        elif pose == 'up-spine':
            return 'up'
        else:
            raise ex("ERROR: Should never be here")
    elif action == "flip":
        if pose == 'down':
            return 'up'
        elif pose == 'up':
            return 'down'
        elif pose == 'down-spine':
            return 'up-spine'
        elif pose == 'up-spine':
            return 'down-spine'
        else:
            raise ex("ERROR: Should never be here")
    elif action == "flip-rotate":
        if pose == 'down':
            return 'up-spine'
        elif pose == 'up':
            return 'down-spine'
        elif pose == 'down-spine':
            return 'up'
        elif pose == 'up-spine':
            return 'down'
        else:
            raise ex("ERROR: Should never be here")
    else:
        raise ex("ERROR: Should never be here")


@memorize
def prevPose(pose, action):
    # these actions pose combinations just happen to be complementary so
    # its the same thing...
    if pose not in poses:
        raise ex("ERROR: unrecognized pose")
    if action not in actions:
        raise ex("ERROR: unrecognized action")
    if action == 'stay':
        return pose
    elif action == "rotate":
        if pose == 'down':
            return 'down-spine'
        elif pose == 'down-spine':
            return 'down'
        elif pose == 'up':
            return 'up-spine'
        elif pose == 'up-spine':
            return 'up'
        else:
            raise ex("ERROR: Should never be here")
    elif action == "flip":
        if pose == 'down':
            return 'up'
        elif pose == 'up':
            return 'down'
        elif pose == 'down-spine':
            return 'up-spine'
        elif pose == 'up-spine':
            return 'down-spine'
        else:
            raise ex("ERROR: Should never be here")
    elif action == "flip-rotate":
        if pose == 'down':
            return 'up-spine'
        elif pose == 'up':
            return 'down-spine'
        elif pose == 'down-spine':
            return 'up'
        elif pose == 'up-spine':
            return 'down'
        else:
            raise ex("ERROR: Should never be here")
    else:
        raise ex("ERROR: Should never be here")


@memorize
def nextPoseIdx(poseIdx, actionIdx):
    return pose.index(nextPose(poses[poseIdx], actions[actionIdx]))


@memorize
def prevPoseIdx(poseIdx, actionIdx):
    return pose.index(prevPose(poses[poseIdx], actions[actionIdx]))


@memorize
def pose2idx(pose):
    return poses.index(pose)


@memorize
def obj2idx(obj):
    return objects.index(obj)


@memorize
def action2idx(action):
    return actions.index(action)


#
# TRAIN
#
class Distribution1D:

    """1D Gaussian Distribution"""

    def __init__(self, values):
        # learn the distribution
        values = np.array(values)
        self.mu = mean(values)
        self.sigma = sqrt(var(values))

    def pdf(self, values):
        return normpdf(values, self.mu, self.sigma)

    def logpdf(self, value):
        return -log(self.sigma * sqrt(2 * pi)) - (value - self.mu) ** 2 / (2 * self.sigma ** 2)


@memorize
def dfgop(idxObject, idxPose, idxFeature):
    # likelihood distribution
    return Distribution1D(errors[idxObject, idxPose, idxFeature, :])


def train():
    """pre-train and cache all the distributions"""
    # PARALLELIZE
    for n in range(N):
        for i in range(I):
            for m in range(M):
                _ = dfgop(n, i, m)


@memorize
def posterior_op(idxObservation, idxObject, idxPose):
    if (idxObservation == 0):
        return 1. / K
    elif (idxObservation == 1):
        prior = posterior_op(idxObservation - 1,
                             idxObject,
                             idxPose)
        thisLikelihood = likelihood(idxObservation,
                                    idxObject,
                                    idxPose)
        thisEvidence = evidence(idxObservation)
        print prior
        print thisLikelihood
        print thisEvidence
        wait()
        return prior * thisLikelihood / thisEvidence
    else:
        previousAction = actionHistory[idxObservation - 1]
        previousActionIdx = action2idx(previousAction)
        previousPoseIdx = prevPoseIdx(idxPose,
                                      previousActionIdx)
        lastPosterior = posterior_op(idxObservation - 1,
                                     idxObject,
                                     previousPoseIdx)
        thisLikelihood = likelihood(idxObservation,
                                    idxObject,
                                    idxPose)
        thisEvidence = evidence(idxObservation)
        return lastPosterior * thisLikelihood / thisEvidence


@memorize
def likelihood(idxObservation, idxObject, idxPose):
    observation = observationHistory[idxObservation]
    if len(observation) != M:
        raise ex(
            "ERROR: Observation length != number of features in the model")
    # PARALLELIZE
    # independent features assumption leads to a product of their probabilities
    accumulate = 1
    for idxFeature in range(M):
        pdf = dfgop(idxObject,
                    idxPose,
                    idxFeature).pdf(observation[idxFeature])
        accumulate = accumulate * pdf
        print accumulate
    return accumulate


@memorize
def evidence(idxObservation):
    accumulate = 0
    if (idxObservation == 1):
        # PARALLELIZE
        for idxObject in range(N):
            # sum over objects
            for idxPose in range(I):
                thisLikelihood = likelihood(idxObservation,
                                            idxObject,
                                            idxPose)
                accumulate = accumulate + thisLikelihood

        prior = posterior_op(idxObservation - 1,
                             idxObject,
                             idxPose)
        accumulate = accumulate * prior
    else:
        # PARALLELIZE
        for idxObject in range(N):
            # sum over objects
            for idxPose in range(I):
                # sum over poses
                previousAction = actionHistory[idxObservation - 1]
                previousActionIdx = action2idx(previousAction)
                previousPoseIdx = prevPoseIdx(idxPose,
                                              previousActionIdx)
                lastPosterior = posterior_op(idxObservation - 1,
                                             idxObject,
                                             previousPoseIdx)
                thisLikelihood = likelihood(idxObservation,
                                            idxObject,
                                            idxPose)
                accumulate = accumulate + lastPosterior * thisLikelihood
    return accumulate


def observe(F):
    """This method gets invokes upon observing data"""
    global observationHistory
    observationHistory.append(F)
    idxObservation = whichObservationIdx()
    # compute and cache the posteriors
    # PARALLELIZE
    for n in range(N):
        for i in range(I):
            posterior_op(idxObservation, n, i)

    # determine optimal action
    # take that action
    # plotPosterior(idxObservation)


def plotPosterior(idxObservation):
    posteriors = array([[posterior_op(idxObservation, n, i)
                        for i in range(I)]
                       for n in range(N)])

    xn, yn = posteriors.shape
    ax.plot_wireframe(
        array([range(xn)] * yn).T, array([range(yn)] * xn), posteriors)
    show()
    wait()

importData()
train()

#
# Test with a training sample
#

oi = 0
pi = 1
r = 0

data = errors[oi, pi, :, r]
observe(data)
