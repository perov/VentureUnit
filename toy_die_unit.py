from venture import shortcuts
ripl = shortcuts.make_church_prime_ripl()

from venture_unit import *

uncollapsed = "uncollapsed"
collapsed = "collapsed"
collapsed_opt = "collapsed_opt"

class ToyDie(VentureUnit):
    def makeAssumes(self):
        self.assume("alpha", "(+ (gamma 1.0 1.0) 0.1)")
        self.assume("make_uncollapsed", "(lambda (alp K) ((lambda (weight) (lambda () (categorical weight))) (symmetric_dirichlet alp K)))")
        self.assume("make_collapsed", "(lambda (alp K) (symmetric_dirichlet_multinomial_make alp K))")
        
        T = self.parameters['topics']
        
        variant_engine = self.parameters['model']
        
        if variant_engine == uncollapsed:
            self.assume("toy", "(make_uncollapsed alpha " + str(T) + ")")
        elif variant_engine == collapsed:
            self.assume("toy", "(make_collapsed alpha " + str(T) + ")")
        elif variant_engine == collapsed_opt:
            self.assume("fast_calc_joint_prob", "1")
            self.assume("toy", "(make_collapsed alpha " + str(T) + ")")
    
    def makeObserves(self):
        W = self.parameters['words']
        for i in range(W):
            self.observe("(toy)", 0)

parameters = {'topics': range(10, 101, 10), 'words': 50, 'model': [collapsed, uncollapsed]}
runner = lambda params : ToyDie(ripl, params).runFromJoint(sweeps=10, runs=1, track=0)
histories = produceHistories(parameters, runner, verbose=True)
plotAsymptotics(parameters, histories, 'sweep_time', aggregate=True)

