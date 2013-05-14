
from venture import engine as MyRIPL

from venture_unit import *

class LDA(VentureUnit):
    def makeAssumes(self):
        self.assume("topics", str(self.parameters['topics']))
        self.assume("vocab", str(self.parameters['vocab']))
        
        self.assume("alpha-document-topic", "(gamma 1.0 1.0)")
        self.assume("alpha-topic-word", "(gamma 1.0 1.0)")
      
        self.assume("get-document-topic-sampler", "(mem (lambda (doc) (symmetric-dirichlet-multinomial/make alpha-document-topic topics)))")
        self.assume("get-topic-word-sampler", "(mem (lambda (topic) (symmetric-dirichlet-multinomial/make alpha-topic-word vocab)))")
        
        self.assume("get-word", "(mem (lambda (doc pos) ((get-topic-word-sampler ((get-document-topic-sampler doc))))))")
        
    def makeObserves(self):
        D = self.parameters['documents']
        N = self.parameters['words_per_document']
        
        for doc in xrange(D):
            for pos in xrange(N):
                self.observe("(get-word %d %d)" % (doc, pos), 0)

#parameters = {'topics' : 4, 'vocab' : 10, 'documents' : 8, 'words_per_document' : 12}
#model = LDA(MyRIPL, parameters)

#history = model.runConditionedFromPrior(50)
#history = model.runFromJoint(50)
#history = model.sampleFromJoint(50)
#history = model.computeJointKL(200, 200, verbose=True)[2]
#history.plot(fmt='png')

parameters = {'topics' : [4, 8], 'vocab' : 10, 'documents' : [8, 12], 'words_per_document' : [4*x for x in range(2, 10)]}
runner = lambda params : LDA(MyRIPL, params).runConditionedFromPrior(sweeps=20, runs=1)
histories = produceHistories(parameters, runner)
plotAsymptotics(parameters, histories, 'sweep_time', aggregate=True)
