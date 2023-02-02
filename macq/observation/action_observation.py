from warnings import warn
from typing import Set
from ..utils import PercentError
from ..trace import Step, Fluent
from ..trace import PartialState
from . import Observation, InvalidQueryParameter


class ActionObservation(Observation):
    """The Action Observability Token.
    Does not store stateful information for use in LOCM suite algorithms
    """

    def __init__(
        self, step: Step,
    ):
        """
        Creates a PartialObservation object, storing the step.
        Args:
            step (Step):
                The step associated with this observation.
            percent_missing (float):
                The percentage of fluents to randomly hide in the observation.
            hide (Set[Fluent]):
                The set of fluents to explicitly hide in the observation.
        """

        # NOTE: Can't use super due to multiple inheritence (NoisyPartialObservation)
        Observation.__init__(self, index=step.index)

        # If percent_missing == 1 -> self.state = None (below).
        # This allows ARMS (and other algorithms) to skip steps when there is no
        # state information available without having to check every mapping in
        # the state (slow in large domains).
        self.state = None
        self.action = None if step.action is None else step.action.clone()

    def __eq__(self, other):
        return (
            isinstance(other, ActionObservation)
            and self.state == None #
            and self.action == other.action
        )

    def hide_random_subset(self, step: Step, percent_missing: float):
        """Hides a random subset of the fluents in the step.
        Args:
            step (Step):
                The step to tokenize.
            percent_missing (float):
                The percentage of fluents to hide (0-1).
        Returns:
            A Step whose state is a PartialState with the random fluents hidden.
        """
        new_fluents = {}
        hidden_f = self.extract_fluent_subset(step.state, percent_missing)
        for f in step.state:
            new_fluents[f] = None if f in hidden_f else step.state[f]
        return Step(PartialState(new_fluents), step.action, step.index)

    def hide_subset(self, step: Step, hide: Set[Fluent]):
        """Hides the specified set of fluents in the observation.
        Args:
            step (Step):
                The step to tokenize.
            hide (Set[Fluent]):
                The set of fluents that will be hidden.
        Returns:
            A Step whose state is a PartialState with the specified fluents hidden.
        """
        new_fluents = {}
        for f in step.state.fluents:
            new_fluents[f] = None if f in hide else step.state[f]
        return Step(PartialState(new_fluents), step.action, step.index)

    def _matches(self, key: str, value: str):
        if key == "action":
            if self.action is None:
                return value is None
            return self.action.details() == value
        elif key == "fluent_holds":
            if self.state is None:
                return value is None
            return self.state.holds(value)
        else:
            raise InvalidQueryParameter(ActionObservation, key)
