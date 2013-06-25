from venture import shortcuts
ripl = shortcuts.make_church_prime_ripl()

from venture_unit import *

class LDA(VentureUnit):
    def makeAssumes(self):
        self.assume("topics", self.parameters['topics'])
        self.assume("vocab", self.parameters['vocab'])
        
        self.assume("alpha_document_topic", "(gamma 1.0 1.0)")
        self.assume("alpha_topic_word", "(gamma 1.0 1.0)")
        
        self.assume("get_document_topic_sampler", "(mem (lambda (doc) (symmetric_dirichlet_multinomial_make alpha_document_topic topics)))")
        self.assume("get_topic_word_sampler", "(mem (lambda (topic) (symmetric_dirichlet_multinomial_make alpha_topic_word vocab)))")
        
        self.assume("get_word", "(mem (lambda (doc pos) ((get_topic_word_sampler ((get_document_topic_sampler doc))))))")
        
    def makeObserves(self):
        D = self.parameters['documents']
        N = self.parameters['words_per_document']
        
        for doc in range(D):
            for pos in range(N):
                self.observe("(get_word %d %d)" % (doc, pos), 0)

#parameters = {'topics' : 4, 'vocab' : 10, 'documents' : 8, 'words_per_document' : 12}
#model = LDA(ripl, parameters)

#history = model.runConditionedFromPrior(50, verbose=True)
#history = model.runFromJoint(50, verbose=True)
#history = model.sampleFromJoint(20, verbose=True)
#history = model.computeJointKL(200, 200, verbose=True)[2]
#history.plot(fmt='png')

parameters = {'topics' : [4, 8], 'vocab' : 10, 'documents' : [8, 12], 'words_per_document' : [4*x for x in range(2, 10)]}
runner = lambda params : LDA(ripl, params).runConditionedFromPrior(sweeps=20, runs=1)
histories = produceHistories(parameters, runner)
plotAsymptotics(parameters, histories, 'sweep_time', aggregate=True)

