'''
Base class of the search slot: which boundaries are tried.
'''
from deltas.core.component import Component


class CandidateSet(Component):
    def generate(self, data, bounds):
        '''array of boundaries to try'''
        raise NotImplementedError
