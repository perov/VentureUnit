from venture import shortcuts
ripl = shortcuts.make_church_prime_ripl()

from venture_unit import *

uncollapsed = 0
collapsed = 1

class ToyDie(VentureUnit):
    def makeAssumes(self):
        self.assume("alpha", "(+ (gamma 1.0 1.0) 0.1)")
        self.assume("make_uncollapsed", "(lambda (alp K) ((lambda (weight) (lambda () (categorical weight))) (symmetric_dirichlet alp K)))")
        self.assume("make_collapsed", "(lambda (alp K) (symmetric_dirichlet_multinomial_make alp K))")
        
        T = self.parameters['T']
        
        variant_engine = self.parameters['variant_engine']
        
        if self.parameters['variant_engine'] == "VentureCollapsedOpt":
            self.assume("fast_calc_joint_prob", "1")
        
        if self.parameters['variant_engine'] == uncollapsed:
            self.assume("toy", "(make_uncollapsed alpha " + str(T) + ")")
        elif self.parameters['variant_engine'] == collapsed:
            self.assume("toy", "(make_collapsed alpha " + str(T) + ")")
        elif self.parameters['variant_engine'] == "VentureCollapsedOpt":
            self.assume("toy", "(make_collapsed alpha " + str(T) + ")")
        else:
            print variant_engine
            raise Exception()
    
    def makeObserves(self):
        W = self.parameters['W']
        for i in range(W):
            self.observe("(toy)", 0)

parameters = {'T': range(10, 101, 10), 'W': 50, 'variant_engine': [collapsed, uncollapsed]}
runner = lambda params : ToyDie(ripl, params).runFromJoint(sweeps=10, runs=1, verbose=True)
histories = produceHistories(parameters, runner)
plotAsymptotics(parameters, histories, 'sweep_time', aggregate=True)

