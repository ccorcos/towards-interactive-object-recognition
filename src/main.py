
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
from mpl_toolkits.mplot3d import Axes3D
from memoize import *

from utils import *
import pprint

# trainingData = {"objectname":{"posename":[feature][training sample]}}
# trainingErrors = [objects][poses][features][training samples]
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
trainingErrors = []
N = I = J = K = M = R = rCrossVal = 0

# FIX: Hard coded actions
actions = ['stay', 'rotate', 'flip', 'flip-rotate']

# first observation at t = 1 = len(observationHistory)-1
# t = 0 represents the prior
observationHistory = [[]]  # no observation or action at t=0
actionHistory = [[]]  # no observation or action at t=0


def clearHistory():
    global observationHistory, actionHistory
    observationHistory = [[]]
    actionHistory = [[]]
    # clearAllMemoized()
    logPosterior_op.reset()
    logLikelihood.reset()
    logEvidence.reset()
    posterior_op.reset()
    likelihood.reset()
    evidence.reset()


def whichObservationIdx():
    return len(observationHistory) - 1


def importTest():
    print "importing test feature data"
    f = open("real_exp.txt", 'r')
    string = f.read()
    data = [s.split('\t')
                    for s in string.split('\n')[:-2]]  # array[feature][sample]
    data = array([[float(s) for s in a] for a in data])
    return data


def importData():
    """Imports SIFT feature errors to training data. Also builds the
    pgm struture with objects and poses arrays"""
    global objects, poses, trainingData
    print "importing sift feature data"
    f = open("MODEL_SIFT_STANDARD2.txt", 'r')
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

    global N, I, J, K, M, R, rCrossVal, trainingErrors, crossValErrors
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
    _, _, M, rErrors = errors.shape

    # cross validation
    rCrossVal = int(round(0.2 * rErrors))
    crossValErrors = errors[:, :,:, :rCrossVal]

    trainingErrors = errors[:, :,:, rCrossVal:]
    _, _, M, R = trainingErrors.shape

    K = N * I

    print "import complete:"
    print str(N) + " objects: "
    pprint.pprint(objects)
    print str(I) + " poses: "
    pprint.pprint(poses)
    print str(M) + " features"
    print str(R) + " training samples"
    print str(rCrossVal) + " cross validation samples"


@memoize
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


@memoize
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


@memoize
def nextPoseIdx(poseIdx, actionIdx):
    return pose.index(nextPose(poses[poseIdx], actions[actionIdx]))


@memoize
def prevPoseIdx(poseIdx, actionIdx):
    return pose.index(prevPose(poses[poseIdx], actions[actionIdx]))


@memoize
def pose2idx(pose):
    return poses.index(pose)


@memoize
def obj2idx(obj):
    return objects.index(obj)


@memoize
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
        # set a minimum
        minlogpdf = log(0.05)
        logpdf = -log(self.sigma * sqrt(2 * pi)) - \
                      (value - self.mu) ** 2 / (2 * self.sigma ** 2)
        return max(minlogpdf, logpdf)


@memoize
def dfgop(idxObject, idxPose, idxFeature):
    # likelihood distribution
    return Distribution1D(trainingErrors[idxObject, idxPose, idxFeature, :])


def train():
    """pre-train and cache all the distributions"""
    # PARALLELIZE
    for n in range(N):
        for i in range(I):
            for m in range(M):
                _ = dfgop(n, i, m)


def plotTraining(idxObject, idxPose):
    print "plotting training for " + objects[idxObject] + " - " + poses[idxPose]
    cm = get_cmap('gist_rainbow')
    x = np.linspace(0, 600, 1000)
    for idxFeature in range(M):
        color = cm(1. * idxFeature / M)
        dist = dfgop(idxObject, idxPose, idxFeature)
        title(objects[idxObject] + " - " + poses[idxPose])
        plot(x, dist.pdf(x))
    show()


@memoize
def logPosterior_op(idxObservation, idxObject, idxPose):
    if (idxObservation == 0):
        return 1. / K
    elif (idxObservation == 1):
        logPrior = logPosterior_op(idxObservation - 1,
                                   idxObject,
                                   idxPose)
        thisLogLikelihood = logLikelihood(idxObservation,
                                          idxObject,
                                          idxPose)
        thisLogEvidence = logEvidence(idxObservation)
        # print prior
        # print thisLogLikelihood
        # print thisLogEvidence
        # wait()
        return logPrior + thisLogLikelihood - thisLogEvidence
    else:
        previousAction = actionHistory[idxObservation - 1]
        previousActionIdx = action2idx(previousAction)
        previousPoseIdx = prevPoseIdx(idxPose,
                                      previousActionIdx)
        lastPosterior = logPosterior_op(idxObservation - 1,
                                        idxObject,
                                        previousPoseIdx)
        thisLogLikelihood = logLikelihood(idxObservation,
                                          idxObject,
                                          idxPose)
        thisLogEvidence = logEvidence(idxObservation)
        return logPrior + thisLogLikelihood - thisLogEvidence


@memoize
def logLikelihood(idxObservation, idxObject, idxPose):
    observation = observationHistory[idxObservation]
    if len(observation) != M:
        raise ex(
            "ERROR: Observation length != number of features in the model")
    # PARALLELIZE
    # independent features assumption leads to a product of their probabilities
    accumulate = 0
    for idxFeature in range(M):
        logpdf = dfgop(idxObject,
                       idxPose,
                       idxFeature).logpdf(observation[idxFeature])
        accumulate = accumulate + logpdf
    return accumulate


def logOfSumGivenLogs(aLogs):
    logC = max(aLogs)
    return log(sum([exp(logA - logC) for logA in aLogs])) + logC


@memoize
def logEvidence(idxObservation):
    # The Trick:
    # log (a+ b) = log (a/c + b/c) + log c
    # c = max(a, b)

    logTerms = []
    if (idxObservation == 1):
        # PARALLELIZE
        for idxObject in range(N):
            # sum over objects
            for idxPose in range(I):
                thisLogLikelihood = logLikelihood(idxObservation,
                                                  idxObject,
                                                  idxPose)
                logPrior = logPosterior_op(idxObservation - 1,
                                           idxObject,
                                           idxPose)

                logTerms.append(thisLogLikelihood + logPrior)
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
                logLastPosterior = logPosterior_op(idxObservation - 1,
                                                   idxObject,
                                                   previousPoseIdx)
                thisLogLikelihood = logLikelihood(idxObservation,
                                                  idxObject,
                                                  idxPose)
                logTerms.append(thisLogLikelihood + logPrior)

    return logOfSumGivenLogs(logTerms)


@memoize
def posterior_op(idxObservation, idxObject, idxPose):
    return exp(logPosterior_op(idxObservation, idxObject, idxPose))


@memoize
def likelihood(idxObservation, idxObject, idxPose):
    return exp(logLikelihood(idxObservation, idxObject, idxPose))


@memoize
def evidence(idxObservation):
    return exp(logEvidence(idxObservation))


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
    ps = posteriors(idxObservation)
    xn, yn = ps.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title("Observation: " + str(idxObservation))
    ax.plot_wireframe(
        array([range(xn)] * yn).T, array([range(yn)] * xn), ps)
    xticks(range(N), objects)
    yticks(range(I), poses)
    show()


def posteriors(idxObservation):
    return array([[posterior_op(idxObservation, n, i)
                   for i in range(I)]
                  for n in range(N)])


def plotPosteriors(ps, objectName="", poseName=""):
    xn, yn = ps[0].shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title(objectName + " - " + poseName)

    cm = get_cmap('gist_rainbow')

    for i in range(len(ps)):
        posterior = ps[i]
        color = cm(1. * i / len(ps))
        # shift a little bit do you can see how many there are
        ax.plot_wireframe(array([range(xn)] * yn).T + 0.01 * i,
                          array([range(yn)] * xn) + 0.01 * i,
                          posterior,
                          color=color)
    xticks(range(N), objects)
    yticks(range(I), poses)
    show()


def plotTrainingPosteriors():
    for n in range(N):
        for i in range(I):
            ps = []
            for r in range(R):
                print "object: " + str(n) + ", pose: " + str(i) + ", sample: " + str(r)
                data = trainingErrors[n, i, :, r]
                observe(data)
                ps.append(posteriors(1))
                clearHistory()
            plotPosteriors(ps, objects[n], poses[i])
            wait()


def plotCrossValPosteriors():
    for n in range(N):
        for i in range(I):
            ps = []
            for r in range(rCrossVal):
                print "object: " + str(n) + ", pose: " + str(i) + ", sample: " + str(r)
                data = crossValErrors[n, i, :, r]
                observe(data)
                ps.append(posteriors(1))
                clearHistory()
            plotPosteriors(ps, objects[n], poses[i])
            wait()

importData()
train()
# plotTraining(0, 0)
# plotTrainingPosteriors()
# plotCrossValPosteriors()

test = importTest()
test1 = test[:, 0]
test2 = test[:, 1]
observe(test1)
plotPosterior(1)
clearHistory()
observe(test2)
plotPosterior(1)
