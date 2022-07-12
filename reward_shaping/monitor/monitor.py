from abc import abstractmethod, ABC
from typing import Callable

import numpy as np

from reward_shaping.monitor.formula import Operator

""" 
Conventions:
    - 2-states automaton: state 0 is unsat (safety violation, target not reached, ..), 1 is sat (safe, target reached)
"""

class GenericMonitor(ABC):
    """ monitor implemented as a Moore machine with states (id,reg) where id is the id state,
    reg is a shared register with quantitative sat measure """

    @abstractmethod
    def __init__(self, predicate: Callable):
        self._p = predicate
        self._terminal_ids = None
        self._counter = None
        self._state_id = None
        self._last_robustness = None
        self._states = None

    @abstractmethod
    def step(self, state):
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    def n_states(self):
        return len(self._states)

    def is_sat(self) -> bool:
        return self._state_id in self._terminal_ids

    def get_counter(self) -> bool:
        return self._counter

    def get_last_robustness(self) -> bool:
        return self._last_robustness


class EnsureMonitor(GenericMonitor):
    def __init__(self, predicate: Callable):
        self._p = predicate
        self._states = {0: "unsafe", 1: "safe"}
        self._terminal_ids = {1}
        self._state_id = None
        self._counter = None
        self._last_robustness = None
        self.reset()

    def reset(self):
        self._state_id = 1
        self._counter = 0
        self._last_robustness = None

    def step(self, state, info={}):
        """
        Transition definition (S x B -> S x R):
            state: (1, k), p(s)>0   -> (1, k++)
            state: (1, k), p(s)<0   -> (0, k)
            state: (0, k), *        -> (0, k)
         """
        self._last_robustness = self._p(state, info)
        # transition
        if self._state_id == 1 and self._last_robustness < 0:
            self._state_id = 0
        # update counter
        if self._state_id == 1 and self._last_robustness >= 0:
            self._counter += 1
        assert self._state_id in self._states, f"unexpected state {self._state_id} in a ensure automaton"
        return self._state_id, self._counter, self._last_robustness


class AchieveMonitor(GenericMonitor):
    def __init__(self, predicate: Callable):
        self._p = predicate
        self._states = {0: "not_achieved", 1: "achieved"}
        self._terminal_ids = {1}
        self._state_id = None
        self._counter = None
        self._last_robustness = None
        self.reset()

    def reset(self):
        self._state_id = 0
        self._counter = 0
        self._last_robustness = None

    def step(self, state, info={}):
        """
        Transition definition (S x B -> S x R):
            state: (0, k), p(s)>0   -> (1, k++)
            state: (0, k), p(s)<0   -> (0, k)
            state: (1, k), p(s)>0   -> (1, k++)
            state: (1, k), p(s)<0   -> (1, k)
         """
        self._last_robustness = self._p(state, info)
        # transition
        if self._state_id == 0 and self._last_robustness >= 0:
            self._state_id = 1
        # update counter
        if self._last_robustness >= 0:
            self._counter += 1
        assert self._state_id in self._states, f"unexpected state {self._state_id} in a achieve automaton"
        return self._state_id, self._counter, self._last_robustness


class ConquerMonitor(GenericMonitor):
    def __init__(self, predicate: Callable):
        self._p = predicate
        self._states = {0: "not_achieved", 1: "achieved", 2: "conquer"}
        self._terminal_ids = {2}
        self._state_id = None
        self._counter = None
        self._last_robustness = None
        self.reset()

    def reset(self):
        self._state_id = 0
        self._counter = 0
        self._last_robustness = None

    def step(self, state, info={}):
        """
        Transition definition (S x R x B -> S x R):
            (0, k), p(x)<0 -> (0, k)
            (0, k), p(x)>0 -> (2, k++)
            (1, k), p(x)<0 -> (1, 0)
            (1, k), p(x)>0 -> (2, k++)
            (2, k), p(x)<0 -> (1, 0)
            (2, k), p(x)>0 -> (2, k++)
         """
        self._last_robustness = self._p(state, info)
        # transition
        if self._state_id == 0 and self._last_robustness >= 0:
            self._state_id = 2
        elif self._state_id == 1 and self._last_robustness >= 0:
            self._state_id = 2
        elif self._state_id == 2 and self._last_robustness < 0:
            self._state_id = 1
        # counter
        if self._last_robustness >= 0:
            self._counter += 1
        else:
            self._counter = 0  # reset counter
        assert self._state_id in self._states, f"unexpected state {self._state_id} in a conquer automaton"
        return self._state_id, self._counter, self._last_robustness


class EncourageMonitor(GenericMonitor):
    def __init__(self, predicate: Callable):
        self._p = predicate
        self._states = {0: "uncomfortable", 1: "comfortable"}
        self._terminal_ids = {0, 1}  # comfort is always satisfied
        self._state_id = None
        self._counter = None
        self._last_robustness = None
        self.reset()

    def reset(self):
        self._state_id = 0
        self._counter = 0
        self._last_robustness = None

    def step(self, state, info={}):
        """
        Transition definition (S x B ->  (S x R):
            state: (*, k), p(s)>0   -> (1, k++)
            state: (*, k), p(s)<0   -> (0, k)
         """
        self._last_robustness = self._p(state, info)
        # transition
        if self._last_robustness >= 0:
            self._state_id = 1
        elif self._last_robustness < 0:
            self._state_id = 0
        # update counter
        if self._last_robustness >= 0:
            self._counter += 1
        assert self._state_id in self._states, f"unexpected state {self._state_id} in a encourage automaton"
        return self._state_id, self._counter, self._last_robustness


class Monitor(GenericMonitor):
    @staticmethod
    def from_spec(operator: Operator, predicate: Callable):
        if operator == Operator.ENSURE:
            return EnsureMonitor(predicate)
        if operator == Operator.ACHIEVE:
            return AchieveMonitor(predicate)
        if operator == Operator.CONQUER:
            return ConquerMonitor(predicate)
        if operator == Operator.ENCOURAGE:
            return EncourageMonitor(predicate)
