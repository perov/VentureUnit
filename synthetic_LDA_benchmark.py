from venture import engine as MyRIPL

#from venture import client
#MyRIPL = client.RemoteRIPL('127.0.0.1', 8082)

from benchmarking import *

class LDA(Benchmarker):
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

parameters = {'topics' : 4, 'vocab' : 10, 'documents' : 8, 'words_per_document' : 12}
lda = LDA(MyRIPL, parameters)

history = lda.runConditionedFromPrior(50)
#history = lda.runFromJoint(50)
#history = lda.sampleFromJoint(50)
#history = lda.computeJointKL(200, 200, verbose=True)[2]
history.plot(fmt='png')
