# simple_rl imports.
from simple_rl.mdp import MDP

class MDPHierarchy(MDP):

	def __init__(self, mdp, sa_stack, aa_stack, sample_rate=10):
		'''
		Args:
			mdp
			sa_stack
			aa_stack
			sample_rate
		'''
		self.mdp = mdp
		self.sa_stack = sa_stack
		self.aa_stack = aa_stack
		self.sample_rate = sample_rate
		
