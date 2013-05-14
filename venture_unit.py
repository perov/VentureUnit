import time
import random
import pdb
from venture.lisp_parser import parse
import numpy

# whether to record a value returned from the RIPL
def record(value):
    if type(value) == bool:
        return True
    elif type(value) == int:
        return True
    elif type(value) == float:
        return True
    elif type(value) == str:
        sep = '['
        if sep in value:
            typeName = value.partition('[')[0]
            if typeName == 'a':
                return True
            else: # can't convert this type to a number
                return False
        else: # probably a LAMBDA or other procedure
            return False
    
    # this probably should not happen
    return False

# Converts venture types to numbers.
# For example, an atom 'a[5]' would become the integer 5.
def parseValue(value):
    if type(value) == bool:
        return value
    elif type(value) == int:
        return value
    elif type(value) == float:
        return value
    elif type(value) == str:
        sep = '['
        if sep in value:
            typeName, bracket, data = value.partition('[')
            data = data[:-1]
            if typeName == 'a':
                return int(data)
            else: # can't convert this type to a number
                return value
        else: # probably a LAMBDA or other procedure
            return value
    
    # this probably should not happen
    return None

# VentureUnit is an experimental harness for developing, debugging and profiling Venture programs.
class VentureUnit:
    RIPL = None
    parameters = {}
    assumes = []
    observes = []
    
    # Register an assume.
    def assume(self, symbol, expression):
        self.assumes.append((symbol, expression))
    
    # Override to create generative model.
    def makeAssumes(self): pass
    
    # Register an observe.
    def observe(self, expression, literal):
        self.observes.append((expression, literal))
    
    # Override to constrain model on data.
    def makeObserves(self): pass
    
    # Initializes parameters, generates the model, and prepares the RIPL.
    def __init__(self, RIPL, parameters={}):
        self.RIPL = RIPL
        
        # FIXME: Should the random seed be stored, or re-initialized?
        self.parameters = parameters.copy()
        if 'venture_random_seed' not in self.parameters:
            self.parameters['venture_random_seed'] = self.RIPL.get_seed()
        else:
            self.RIPL.set_seed(self.parameters['venture_random_seed'])
        
        # FIXME: automatically assume parameters (and omit them from history)?
        self.assumes = []
        self.makeAssumes()
        
        self.observes = []
        self.makeObserves()
    
    # Loads the assumes and changes the observes to predicts.
    # Also picks a subset of the predicts to track (by default all are tracked).
    # Prunes non-scalar values, unless prune=False.
    # Does not reset engine RNG.
    def loadModelWithPredicts(self, track=-1, prune=True):
        self.RIPL.clear()
        
        assumeToDirective = {}
        for (symbol, expression) in self.assumes:
            (directive, value) = self.RIPL.assume(symbol, parse(expression))
            if (not prune) or record(value):
                assumeToDirective[symbol] = directive
        
        predictToDirective = {}
        for (index, (expression, literal)) in enumerate(self.observes):
            (directive, value) = self.RIPL.predict(parse(expression))
            if (not prune) or record(value):
                predictToDirective[index] = directive
        
        # choose a random subset to track, by default all are tracked
        if track >= 0:
            track = min(track, len(predictToDirective))
            # FIXME: need predictable behavior from RNG
            random.seed(parameters['venture_random_seed'])
            predictToDirective = dict(random.sample(predictToDirective.items(), track))
        
        return (assumeToDirective, predictToDirective)
    
    # Updates recorded values after an iteration of the RIPL.
    def updateValues(self, keyedValues, keyToDirective):
        for (key, values) in keyedValues.items():
            if key not in keyToDirective: # we aren't interested in this series
                del keyedValues[key]
                continue
            
            value = self.RIPL.report_value(keyToDirective[key])
            if len(values) > 0:
                if type(value) == type(values[0]):
                    values.append(value)
                else: # directive has returned a different type; discard the series
                    del keyedValues[key]
            elif record(value):
                values.append(value)
            else: # directive has returned a non-scalar type; discard the series
                del keyedValues[key]

    # Gives a name to an observe directive.
    def nameObserve(self, index):
        return 'observe[' + str(index) + '] ' + self.observes[index][0]
    
    # Provides independent samples from the joint distribution (observes turned into predicts).
    # A random subset of the predicts are tracked along with the assumed variables.
    def sampleFromJoint(self, samples, track=5, verbose=False):
        assumedValues = {}
        for (symbol, expression) in self.assumes:
          assumedValues[symbol] = []
        predictedValues = {}
        for index in range(len(self.observes)):
          predictedValues[index] = []
        
        logscores = []
        
        for i in range(samples):
            if verbose:
                print "Generating sample " + str(i)
            
            (assumeToDirective, predictToDirective) = self.loadModelWithPredicts(track)
            
            logscores.append(self.RIPL.logscore())
            
            self.updateValues(assumedValues, assumeToDirective)
            self.updateValues(predictedValues, predictToDirective)
        
        history = History('sample_from_joint', self.parameters)
        
        history.addSeries('logscore', 'i.i.d.', logscores)
        
        series = assumedValues.copy()
        for (symbol, values) in assumedValues.iteritems():
            history.addSeries(symbol, 'i.i.d.', map(parseValue, values))
        
        for (index, values) in predictedValues.iteritems():
            history.addSeries(self.nameObserve(index), 'i.i.d.', map(parseValue, values))
        
        return history
    
    # iterates until (approximately) all random choices have been resampled
    def sweep(self):
        iterations = 0
        
        while iterations < self.RIPL.get_entropy_info()['unconstrained_random_choices']:
            step = self.RIPL.get_entropy_info()['unconstrained_random_choices']
            self.RIPL.infer(step)
            iterations += step
        
        return iterations
    
    # Runs inference on the joint distribution (observes turned into predicts).
    # A random subset of the predicts are tracked along with the assumed variables.
    def runFromJoint(self, sweeps, track=5, runs=3, verbose=False):
        history = History('run_from_joint', self.parameters)
        
        for run in range(runs):
            if verbose:
                print "Starting run " + str(run)
            
            (assumeToDirective, predictToDirective) = self.loadModelWithPredicts(track)
            
            assumedValues = {}
            for symbol in assumeToDirective:
              assumedValues[symbol] = []
            predictedValues = {}
            for index in predictToDirective:
              predictedValues[index] = []
            
            sweepTimes = []
            sweepIters = []
            logscores = []
            
            for sweep in range(sweeps):
                if verbose:
                    print "Running sweep " + str(sweep)
                
                # FIXME: use timeit module for better precision
                start = time.clock()
                iterations = self.sweep()
                end = time.clock()
                
                sweepTimes.append(end-start)
                sweepIters.append(iterations)
                logscores.append(self.RIPL.logscore())
                
                self.updateValues(assumedValues, assumeToDirective)
                self.updateValues(predictedValues, predictToDirective)
            
            history.addSeries('sweep_time', 'run ' + str(run), sweepTimes)
            history.addSeries('sweep_iters', 'run ' + str(run), sweepIters)
            history.addSeries('logscore', 'run ' + str(run), logscores)
            
            for (symbol, values) in assumedValues.iteritems():
                history.addSeries(symbol, 'run ' + str(run), map(parseValue, values))
            
            for (index, values) in predictedValues.iteritems():
                history.addSeries(self.nameObserve(index), 'run ' + str(run), map(parseValue, values))
        
        return history
    
    
    # Computes the KL divergence on i.i.d. samples from the prior and inference on the joint.
    # Returns the sampled history, inferred history, and history of KL divergences.
    def computeJointKL(self, sweeps, samples, track=5, runs=3, verbose=False):
        sampledHistory = self.sampleFromJoint(samples, track, verbose)
        inferredHistory = self.runFromJoint(sweeps, track, runs, verbose)
        
        klHistory = History('kl_divergence', self.parameters)
        
        for (name, seriesList) in inferredHistory.nameToSeries.iteritems():
            if name not in sampledHistory.nameToSeries: continue
            
            for inferredSeries in seriesList:
                sampledSeries = sampledHistory.nameToSeries[name][0]
                
                klValues = [computeKL(sampledSeries.values[:index+1], inferredSeries.values) for index in range(sweeps)]
                
                klHistory.addSeries('KL_' + name, inferredSeries.label, klValues, hist=False)
        
        return (sampledHistory, inferredHistory, klHistory)
    
    # Runs inference on the model conditioned on observed data.
    # By default the data is as given in makeObserves(parameters).
    def runFromConditional(self, sweeps, data=None, runs=3, verbose=False):        
        history = History('run_from_conditional', self.parameters)
        
        for run in range(runs):
            if verbose:
                print "Starting run " + str(run)
            
            self.RIPL.clear()
        
            assumeToDirective = {}
            for (symbol, expression) in self.assumes:
                (directive, value) = self.RIPL.assume(symbol, parse(expression))
                if record(value): assumeToDirective[symbol] = directive
        
            for (index, (expression, literal)) in enumerate(self.observes):
                datum = literal if data is None else data[index]
                self.RIPL.observe(parse(expression), datum)

            sweepTimes = []
            sweepIters = []
            logscores = []
            
            assumedValues = {}
            for symbol in assumeToDirective:
              assumedValues[symbol] = []
              
            for sweep in range(sweeps):
                if verbose:
                    print "Running sweep " + str(sweep)
                
                # FIXME: use timeit module for better precision
                start = time.time()
                iterations = self.sweep()
                end = time.time()
                
                sweepTimes.append(end-start)
                sweepIters.append(iterations)
                logscores.append(self.RIPL.logscore())
                
                self.updateValues(assumedValues, assumeToDirective)
            
            history.addSeries('sweep_time', 'run ' + str(run), sweepTimes)
            history.addSeries('sweep_iters', 'run ' + str(run), sweepIters)
            history.addSeries('logscore', 'run ' + str(run), logscores)
            
            for (symbol, values) in assumedValues.iteritems():
                history.addSeries(symbol, 'run ' + str(run), map(parseValue, values))
        
        return history
    
    # Run inference conditioned on data generated from the prior.
    def runConditionedFromPrior(self, sweeps, runs=3, verbose=False):
        if verbose:
            print 'Generating data from prior'
        
        (assumeToDirective, predictToDirective) = self.loadModelWithPredicts(prune=False)
        
        data = [self.RIPL.report_value(predictToDirective[index]) for index in range(len(self.observes))]
        
        assumedValues = {}
        for (symbol, directive) in assumeToDirective.iteritems():
            value = self.RIPL.report_value(directive)
            if record(value):
                assumedValues[symbol] = value
        
        logscore = self.RIPL.logscore()
        
        history = self.runFromConditional(sweeps, data, runs, verbose)
        
        history.addSeries('logscore', 'prior', [logscore]*sweeps, hist=False)
        for (symbol, value) in assumedValues.iteritems():
            history.addSeries(symbol, 'prior', [parseValue(value)]*sweeps)
        
        history.label = 'run_conditioned_from_prior'
        
        return history

from numpy import mean

# Records data for each sweep. Typically, all scalar assumes are recorded.
# Certain running modes convert observes to predicts. In those cases, a random subset of the observes (now predicts) are tracked.
# Some extra data is also recorded, such as the logscore, sweep_time, and sweep_iters.
class History:
    def __init__(self, label='empty_history', parameters={}):
        self.label = label
        self.parameters = parameters
        self.nameToSeries = {}
    
    def addSeries(self, name, label, values, hist=True):
        if name not in self.nameToSeries:
            self.nameToSeries[name] = []
        self.nameToSeries[name].append(Series(label, values, hist))
    
    def averageValue(self, seriesName):
        if seriesName not in self.nameToSeries:
            return None
        
        return [(series.label, mean(series.values)) for series in self.nameToSeries[seriesName]]
    
    # default directory for plots, created from parameters
    def defaultDirectory(self):
        name = self.label
        for (param, value) in self.parameters.iteritems():
            name += '_' + param + '=' + str(value)
        return name + '/'
    
    # directory specifies location of plots
    # default format is pdf
    def plot(self, fmt='pdf', directory=None):
        if directory == None:
            directory = self.defaultDirectory()
        
        if not os.path.exists(directory):
            os.mkdir(directory)
        
        for (name, seriesList) in self.nameToSeries.iteritems():
            plotSeries(name, self.label, seriesList, self.parameters, fmt, directory)
            plotHistogram(name, self.label, seriesList, self.parameters, fmt, directory)
        
        print 'plots written to ' + directory

# aggregates values for one variable over the course of a run
class Series:
    def __init__(self, label, values, hist):
        self.label = label
        self.values = values
        self.hist = hist

import matplotlib
#matplotlib.use('pdf')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
import os

# Displays parameters in top-left corner of the graph.
def showParameters(parameters):
    items = sorted(parameters.items())
    
    text = items[0][0] + ' = ' + str(items[0][1])
    for (name, value) in items[1:]:
        text += '\n' + name + ' = ' + str(value)
    
    plt.text(0, 1, text, transform=plt.axes().transAxes, va='top', size='small', linespacing=1.0)

# Plots a set of series.
def plotSeries(name, subtitle, seriesList, parameters, fmt, directory):
    fig = plt.figure()
    plt.clf()
    plt.title('Series for ' + name + '\n' + subtitle)
    plt.xlabel('Sweep')
    plt.ylabel(name)
    showParameters(parameters)
    
    plots = [plt.plot(series.values)[0] for series in seriesList]
    
    plt.legend(plots, [series.label for series in seriesList])

    ymin = min([min(series.values) for series in seriesList])
    ymax = max([max(series.values) for series in seriesList])

    offset = 0.1 * max([(ymax - ymin), 1.0])

    if not any([any([numpy.isinf(v) for v in series.values]) for series in seriesList]):
        plt.ylim([ymin - offset, ymax + offset])
    
    #plt.tight_layout()
    fig.savefig(directory + name.replace(' ', '_') + '_series.' + fmt, format=fmt)

# Plots histograms for a set of series.
def plotHistogram(name, subtitle, seriesList, parameters, fmt, directory):
    fig = plt.figure()
    plt.clf()
    plt.title('Histogram of ' + name + '\n' + subtitle)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    showParameters(parameters)
    
    # FIXME: choose a better bin size
    plt.hist([series.values for series in seriesList], bins=20, label=[series.label for series in seriesList])
    plt.legend()
    
    #plt.tight_layout()
    fig.savefig(directory + name.replace(' ', '_') + '_hist.' + fmt, format=fmt)

# smooths out a probability distribution function
def smooth(pdf, amt=0.1):
    return [(p + amt / len(pdf)) / (1.0 + amt) for p in pdf]

import numpy as np
#np.seterr(all='raise')
import math

# Approximates the KL divergence between samples from two distributions.
# 'reference' is the "true" distribution
# 'approx' is an approximation of 'reference'
def computeKL(reference, approx, numbins=20):
    
    mn = min(reference + approx)
    mx = max(reference + approx)
    
    refHist = np.histogram(reference, bins=numbins, range = (mn, mx), density=True)[0]
    apxHist = np.histogram(approx, bins=numbins, range = (mn, mx), density=True)[0]
    
    refPDF = smooth(refHist)
    apxPDF = smooth(apxHist)
    
    kl = 0.0
    
    for (p, q) in zip(refPDF, apxPDF):
        kl += math.log(p/q) * p * (mx-mn) / numbins
    
    return kl

import itertools
from collections import namedtuple
from matplotlib import cm

def makeIterable(obj):
    return obj if hasattr(obj, '__iter__') else [obj]

def cartesianProduct(keyToValues):
    items = [(key, makeIterable(value)) for (key, value) in keyToValues.items()]
    
    Key = namedtuple('Key', [key for (key, _) in items])
    
    return [Key._make(t) for t in itertools.product(*[values for (_, values) in items])]

# Produces histories for a set of parameters.
# Here the parameters can contain lists. For example, {'a':[0, 1], 'b':[2, 3]}.
# Then histories will be computed for the parameter settings ('a', 'b') = (0, 1), (0, 2), (1, 2), (1, 3)
# Runner should take a given parameter setting and produce a history.
# For example, runner = lambda params : Model(RIPL, params).runConditionedFromPrior(sweeps, runs)
# Returned is a dictionary mapping each parameter setting (as a namedtuple) to the history.
def produceHistories(parameters, runner):
    returning_dictionary = {}
    for params in cartesianProduct(parameters):
        returning_dictionary[params] = runner(params._asdict())
    return returning_dictionary
    # return {params : runner(params._asdict()) for params in cartesianProduct(parameters)}

def addToDict(dictionary, key, value):
    dictionary[key] = value
    return dictionary

# Produces plots for a a given variable over a set of runs.
# If aggregate=True, multiple plots that differ in only one parameter are overlayed.
def plotAsymptotics(parameters, histories, seriesName, fmt='pdf', directory=None, verbose=False, aggregate=False):
    if directory is None:
        directory = seriesName + '_asymptotics/'
    
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    Key = namedtuple('Key', parameters.keys())
    paramsToValue = {params : mean([value for (_, value) in history.averageValue(seriesName)]) for (params, history) in histories.items()}
    
    for (key, values) in parameters.items():
        if not hasattr(values, '__iter__'):
            continue
        
        others = parameters.copy()
        del others[key]
        
        if aggregate:
            for (other, otherValues) in others.items():
                otherValues = makeIterable(otherValues)
                
                rest = others.copy()
                del rest[other]
                
                for params in cartesianProduct(rest):
                    fig = plt.figure()
                    plt.clf()
                    plt.title(seriesName + ' versus ' + key)
                    plt.xlabel(key)
                    plt.ylabel(seriesName)
                    showParameters(params._asdict())
                    
                    colors = cm.rainbow(np.linspace(0, 1, len(otherValues)))
                    
                    for (otherValue, c) in zip(otherValues, colors):
                        p = addToDict(params._asdict(), other, otherValue)
                        plt.scatter(values, [paramsToValue[Key(**addToDict(p, key, value))] for value in values], label=other+'='+str(otherValue), color=c)
                    
                    plt.legend()
                    
                    filename = key
                    for (param, value) in params._asdict().items():
                        filename += '_' + param + '=' + str(value)
                    
                    #plt.tight_layout()
                    fig.savefig(directory + filename.replace(' ', '_') + '_asymptotics.' + fmt, format=fmt)
        else:
            for params in cartesianProduct(others):
                fig = plt.figure()
                plt.clf()
                plt.title(seriesName + ' versus ' + key)
                plt.xlabel(key)
                plt.ylabel(seriesName)
                showParameters(params._asdict())
                
                plt.scatter(values, [paramsToValue[Key(**addToDict(params._asdict(), key, v))] for v in values])
                
                filename = key
                for (param, value) in params._asdict().items():
                    filename += '_' + param + '=' + str(value)
                
                #plt.tight_layout()
                fig.savefig(directory + filename.replace(' ', '_') + '_asymptotics.' + fmt, format=fmt)

