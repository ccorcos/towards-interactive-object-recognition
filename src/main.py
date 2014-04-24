# -*- coding: utf-8 -*-
'''
Created on 2014-04-23 12:19
@summary: 
@author: chetcorcos

@global objects: array of objectnames
@global poses: array of posenames
@global N: number of objects, len(objects)
@global I: numbher of poses, len(poses)
@global J: number of actions = I
@global K: number of objects*poses, N*I
@global M: number of features
@global Re: total number of samples in fileName
@global R: number of training samples
@global Rx: number of cross-validation samples
@global trainingData: dictionary of training data
    trainingData format:
    {
        'objectname':{
            'posename':[feature1error,feature2error1, ...]
        }
    }
@global trainingErrors: array of training errors 
    [object][pose][feature][sample]
@global crossValErrors: array of cross-validation errors 
    [object][pose][feature][sample]
'''

# pylab imports numpy and matplotlib
from pylab import *

# for 3D plotting of object-pose posterior
from mpl_toolkits.mplot3d import Axes3D

# for caching function calls
from memoize import *

# my own simple utilities
from utils import *


# initialize some global variables
objects = []
poses = []
trainingErrors = []
crossValErrors = []
trainingData = {}
N = 0
I = 0
J = 0
K = 0
M = 0
Re = 0
R = 0
Rx = 0

# FIX: Hard coded actions
actions = ['stay', 'rotate', 'flip', 'flip-rotate']

# first observation at t = 1 = len(observationHistory)-1
# t = 0 represents the prior
observationHistory = [[]]  # no observation or action at t=0
actionHistory = [[]]  # no observation or action at t=0

# ----------------------------------------------------------
# MAIN
# some functions converting actions and poses
# ----------------------------------------------------------

def main():

    trainingFile = "MODEL_SIFT_STANDARD2.txt"
    percentHoldOut = 0.2  # cross-validation
    testFile = "real_exp.txt"

    importTrainingData(trainingFile, percentHoldOut)
    train()

    # PLOT TRAINING
    # If M is large, plotting the training may take a while:
    # with M = 650, it took ~ 20 seconds to render
    # plotFeatureTrainingDistributions()
    # plotObjPoseTrainingDistributions(together=True)
    # plotObjPoseTrainingDistributions(together=False)
    
    # PLOT CROSS VALIDATION
    # plotCrossValPosteriors()

    # PLOT TEST DATA
    # plotTestPosteriors(testFile)
    
    pr(0, "finished")


# ----------------------------------------------------------
# TRAINING
# some functions for training the model
# ----------------------------------------------------------


def importTrainingData(fileName, xValPercent):
    '''
    @summary: Import feature errors for various object-poses,
        defining the object, pose, feature structure of the 
        model

    @param fileName: Name of the file to import.

        fileName format:
        objectname, posename
        feature1error1, feature1error2, feature1error3,...
        feature2error1, feature2error2, feature2error3,...
        ...
        featureNerror1, featureNerror2, featureNerror3,...
        objectname, posename
        ...

    @param xValPercent: percent of training samples held out 
        for cross-validation

    @result: sets the following global variables:
        
        @global objects: array of objectnames
        
        @global poses: array of posenames
        
        @global N: number of objects, len(objects)
        
        @global I: numbher of poses, len(poses)
        
        @global J: number of actions = I
        
        @global K: number of objects*poses, N*I
        
        @global M: number of features

        @global Re: total number of samples in fileName
        
        @global R: number of training samples
        
        @global Rx: number of cross-validation samples
        
        @global trainingData: dictionary of training data

            trainingData format:
            {
                'objectname':{
                    'posename':[feature1error,feature2error1, ...]
                }
            }

        @global trainingErrors: array of training errors 
            [object][pose][feature][sample]
        
        @global crossValErrors: array of cross-validation errors 
            [object][pose][feature][sample]
    '''

    global objects, poses, trainingData, N, I, J, K, M, Re, R, Rx, trainingErrors, crossValErrors

    pr(0, "importing feature errors from", fileName)

    # open file
    f = open(fileName, 'r')
    # read the string
    string = f.read()
    # some simple formatting
    string = string.replace(' ', '').replace(',\n', '\n')
    # split the lines
    lines = string.split('\n')
    # remove the training empty  lines
    while lines[-1] == "":
        lines = lines[:-1]
    # split the numbers in each line
    elements = [x.split(',') for x in lines]

    o = ''
    p = ''
    for line in elements:
        if len(line) == 2:
            # if we are on an "objectname,posename" line
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
            # use a dictionary for simplicity
            # convert string to float
            trainingData[o][p].append([float(i) for i in line])

    N = len(objects)  # number of objects
    I = len(poses)  # number of poses
    J = len(actions)  # number of actions

    # get all of the errors into a list
    # errors.shape will be (N, I, M, R + Rx)
    errors = []
    for n in range(N):
        obj = objects[n]
        errors.append([])
        for i in range(I):
            pose = poses[i]
            errors[n].append(trainingData[obj][pose])
    errors = array(errors)
    # M: number of features
    # Re: number of samples
    _, _, M, Re = errors.shape

    # split the samples into training and cross-validation
    Rx = int(round(xValPercent * Re))
    crossValErrors = errors[:, :,:, :Rx]

    trainingErrors = errors[:, :,:, Rx:]
    _, _, _, R = trainingErrors.shape

    # K: number of object-poses
    K = N * I

    pr(1, "finished import")
    pr(2, "N objects:", N)
    pr(3, objects)
    pr(2, "I poses:", I)
    pr(3, poses)
    pr(2, "M features:", M)
    pr(2, "Re samples:", Re)
    pr(2, "R for training:", R)
    pr(2, "Rx for cross-val:", Rx)


def importTestData(fileName):
    '''
    @summary: import errors for each feature for each observation

    @param fileName:

        fileName format:
        obs1feature1error \t obs2feature1error \t obs3feature1error \t ...
        obs1feature2error \t obs2feature2error \t obs3feature2error \t ...
        obs1feature3error \t obs2feature3error \t obs3feature3error \t ...
        ...
        obs1featureMerror \t obs2featureMerror \t obs3featureMerror \t ...

    @result: data is an array of shape (M, sample or observation)
    '''

    pr(0, "importing test data from", fileName)
    # open file
    f = open(fileName, 'r')
    # read string
    string = f.read()
    lines = string.split('\n')
    # remove the training empty
    while lines[-1] == "":
        lines = lines[:-1]
    # split by tabs
    data = [s.split('\t') for s in lines]  # array[feature][sample]
    # convert string to float and list to array
    data = array([[float(s) for s in a] for a in data])
    return data


class Distribution1D:

    """1D Gaussian Distribution"""

    def __init__(self, values):
        '''
        @summary: creates a 1D gaussian distribution
        @param values: an array of values
        @result: sets instance variables mu for the mean
            and sigma for the standard deviation
        '''

        # learn the distribution
        values = np.array(values)
        self.mu = mean(values)
        self.sigma = sqrt(var(values))

    def pdf(self, values):
        """returns the probability density"""
        return normpdf(values, self.mu, self.sigma)

    def logpdf(self, value):
        """returns the log probability density"""
        return -log(self.sigma * sqrt(2 * pi)) - (value - self.mu) ** 2 / (2 * self.sigma ** 2)


@memoize
def dfgop(idxObject, idxPose, idxFeature):
    """likelihood distribution of p(f|o,p) for the training data"""
    return Distribution1D(trainingErrors[idxObject, idxPose, idxFeature, :])


def train():
    """precomputes all the likelihood distributions"""
    # PARALLELIZE
    for n in range(N):
        for i in range(I):
            for m in range(M):
                _ = dfgop(n, i, m)


# ----------------------------------------------------------
# STATE
# functions for keeping track of state. clearing cache and
# history, observing data, etc.
# ----------------------------------------------------------

def clearHistory():
    '''
    @summary: clears the function call caches
    '''

    global observationHistory, actionHistory
    observationHistory = [[]]
    actionHistory = [[]]

    logPosterior_op.reset()
    logLikelihood.reset()
    logEvidence.reset()
    posterior_op.reset()
    likelihood.reset()
    evidence.reset()


def whichObservationIdx():
    '''
    @summary: to determine which observation we are on, first, second, etc.
    @result: returns the index into the observation history
    '''

    return len(observationHistory) - 1


def observe(F, printing=False):
    '''
    @summary: observe a set of features
    @param F: an array of length M
    @result: sets the global variable for observationHistory
        and precomputes the posteriors
    '''

    global observationHistory
    # keep track of the observation history
    observationHistory.append(F)
    idxObservation = whichObservationIdx()

    if printing:
        pr(0, "observation", idxObservation)

    # compute and cache the posteriors
    # PARALLELIZE
    for n in range(N):
        for i in range(I):
            # pr(2, "posterior:", "o",n,", p",i)
            posterior_op(idxObservation, n, i)


# ----------------------------------------------------------
# INFERENCE
# functions for inferencing on the model
# ----------------------------------------------------------

def logOfSumGivenLogs(aLogs):
    """A nifty trick for taking the sum of really small numbers
    when all you have is the log of those small numbers and you
    want a log of their sum back"""
    logC = max(aLogs)
    return log(sum([exp(logA - logC) for logA in aLogs])) + logC


@memoize
def logPosterior_op(idxObservation, idxObject, idxPose):
    if (idxObservation == 0):
        # posterior 0 is the prior
        return 1. / K
    elif (idxObservation == 1):
        # posterior 1 doesnt take into account any actions yet
        logPrior = logPosterior_op(idxObservation - 1,
                                   idxObject,
                                   idxPose)

        thisLogLikelihood = logLikelihood(idxObservation,
                                          idxObject,
                                          idxPose)

        thisLogEvidence = logEvidence(idxObservation)

        return logPrior + thisLogLikelihood - thisLogEvidence
    else:
        # compute the previous pose
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
        raise ex("ERROR: len(observation) != M")

    # PARALLELIZE
    accumulate = 0
    for idxFeature in range(M):
        logpdf = dfgop(idxObject,
                       idxPose,
                       idxFeature).logpdf(observation[idxFeature])

        accumulate = accumulate + logpdf

    return accumulate


@memoize
def logEvidence(idxObservation):

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


def posteriors_op(idxObservation):
    """returns a matrix of posteriors [object][pose]"""
    return array([[posterior_op(idxObservation, n, i)
                   for i in range(I)]
                  for n in range(N)])


# ----------------------------------------------------------
# ACTIONS
# some functions converting actions and poses
# ----------------------------------------------------------

@memoize
def nextPose(pose, action):
    '''
    @summary: computes the next pose given an action
    @param pose: pose name
    @param action: action name
    @result: next pose name
    '''
    # FIX: hard coded
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
    '''
    @summary: computes the previous pose given an action
    @param pose: pose name
    @param action: action name
    @result: previous pose name
    '''
    # these actions pose combinations just happen to be complementary so
    # its the same thing as next pose
    # FIX: hard coded
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

# ----------------------------------------------------------
# PLOTTING
# some functions plotting training, posteriors, etc.
# ----------------------------------------------------------


def plotPosteriors_op(idxObservation):
    """Plots a 3D wireframe of the object-pose posteriors
    for a specific observation"""

    pr(1, "plotting object-pose posteriors for obs", idxObservation)
    ps = posteriors_op(idxObservation)
    xn, yn = ps.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title("Observation: " + str(idxObservation))
    ax.plot_wireframe(
        array([range(xn)] * yn).T, array([range(yn)] * xn), ps)
    xticks(range(N), objects)
    yticks(range(I), poses)
    show()


def plotPosteriors(ps, objectName="", poseName=""):
    """Plots a 3D wirefram of the object pose posteriors given
    as parameters"""

    pr(1, "plotting object-pose posteriors truth:", objectName, poseName)

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
    """Iterates through the training data plotting the first
    posterior after observing each training data sample"""

    pr(0, "observing and plotting training data")

    for n in range(N):
        for i in range(I):
            ps = []
            for r in range(R):
                data = trainingErrors[n, i, :, r]
                observe(data)
                ps.append(posteriors_op(1))
                clearHistory()
            plotPosteriors(ps, objects[n], poses[i])
            wait()


def plotCrossValPosteriors():
    """Iterated through the cross-validation data plotting
    the first posterior after observing each sample"""

    pr(0, "observing and plotting cross-validation data")

    for n in range(N):
        for i in range(I):
            ps = []
            for r in range(Rx):
                data = crossValErrors[n, i, :, r]
                observe(data)
                ps.append(posteriors_op(1))
                clearHistory()
            plotPosteriors(ps, objects[n], poses[i])
            wait()


def plotObjPoseTrainingDistribution(idxObject, idxPose):
    """Plots the learned feature distributions for a 
    specific object-pose"""

    pr(1, "plotting training for", objects[idxObject], "-", poses[idxPose])

    cm = get_cmap('gist_rainbow')
    x = np.linspace(0, 600, 1000)
    for idxFeature in range(M):
        color = cm(1. * idxFeature / M)
        dist = dfgop(idxObject, idxPose, idxFeature)
        plot(x, dist.pdf(x), color=color)
    title("Training: " + objects[idxObject] + " - " + poses[idxPose])
    show()

def plotObjPoseTrainingDistributions(together=True):
    if together:
        pr(1, "plotting all training distributions together")

        cm = get_cmap('gist_rainbow')
        x = np.linspace(0, 600, 1000)
        for n in range(N):
            for i in range(I):
                for idxFeature in range(M):
                    color = cm(1. * float(n*i+i) / (N*I))
                    dist = dfgop(n, i, idxFeature)
                    plot(x, dist.pdf(x),color=color)
        title("All Training Distributions")
        show()
    else:
        for n in range(N):
            for i in range(I):
                plotObjPoseTrainingDistribution(n,i)
                wait()

def plotFeatureTrainingDistribution(idxFeature):
    """plots the learned distribution a specific feature of all object poses"""
    pr(1, "plotting feature training for feature", idxFeature)

    ax = subplot(1,1,1)
    cm = get_cmap('gist_rainbow')
    x = np.linspace(0, 600, 1000)
    dash = False
    for idxObject in range(N):
        for idxPose in range(I):
            color = cm(1. * float(idxObject*I+idxPose) / (N*I))
            dist = dfgop(idxObject, idxPose, idxFeature)
            l = objects[idxObject] + " - " + poses[idxPose]
            if dash:
                dash = False
                ax.plot(x, dist.pdf(x), '--', label=l, color=color)
            else:
                dash = True
                ax.plot(x, dist.pdf(x), label=l, color=color)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,labels, fontsize=10)
    title("Trained distributions for feature: " + str(idxFeature))
    show()

def plotFeatureTrainingDistributions():
    for idxFeature in range(M):
        plotFeatureTrainingDistribution(idxFeature)
        wait()

def plotTestPosteriors(testFile):

    test = importTestData(testFile)

    m,n = test.shape
    for i in range(n):
        observe(test[:,i])
        plotPosteriors_op(1)
        clearHistory()
        wait()

if __name__ == "__main__":
    main()