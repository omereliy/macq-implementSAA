from enum import Enum
from typing import Union, List, Callable


class CustomObject:
    def __init__(self, name):
        self.name = name


class Fluent:
    def __init__(self, name: str, objects: List[CustomObject], value: bool):
        """
        Class to handle a predicate and the objects it is applied to.

        Arguments
        ---------
        name : str
            The name of the predicate.
        objects : List
            The list of objects this predicate applies to.
        value : bool
            The value of the fluent (true or false)
        """
        self.name = name
        self.objects = objects
        self.value = value


class Effect(Fluent):
    def __init__(
        self,
        name: str,
        objects: List[CustomObject],
        func: Callable,
        probability: int = 100,
    ):
        """
        Class to handle an individual effect of an action.

        Arguments
        ---------
        name : str
            The name of the effect.
        objects : List
            The list of objects this effect applies to.
        func : function
            The function that applies the effect in the corresponding action.
        probability : int
            For non-deterministic problems, the probability that this effect will take place
            (defaults to 100).
        """
        super().__init__(name, objects)
        self.func = func
        self.probability = self.set_prob(probability)

    def set_prob(self, prob: int):
        """
        Setter function for probability.

        Arguments
        ---------
        prob : int
            The probability to be assigned.

        Returns
        -------
        prob : int
            The probability, after being checked for validity.
        """
        # enforce that probability is between 0 and 100 inclusive
        if prob < 0:
            prob = 0
        elif prob > 100:
            prob = 100
        return prob


class Action:
    def __init__(self, name: str, obj_params: List[CustomObject]):
        """
        Class to handle each action.

        Arguments
        ---------
        name : str
            The name of the action.
        obj_params : List
            The list of objects this action applies to.

        Other Class Attributes
        ---------
        precond : List of Fluents
            The list of preconditions needed for this action.
        effects : List of Effects
            The list of effects this action results in/
        """
        self.name = name
        self.obj_params = obj_params
        self.precond = []
        self.effects = []

    def add_effect(
        self,
        name: str,
        objects: List[CustomObject],
        func: Callable,
        probability: int = 100,
    ):
        """
        Creates an effect and adds it to this action.

        Arguments
        ---------
        name : str
            The name of the effect.
        objects : List
            The list of objects this effect applies to.
        func : function
            The function that applies the effect in the corresponding action.
        probability : int
            For non-deterministic problems, the probability that this effect will take place
            (defaults to 100)

        Returns
        -------
        None
        """
        for obj in objects:
            if obj not in self.obj_params:
                raise Exception(
                    "Object must be one of the objects supplied to the action"
                )
        effect = Effect(name, objects, func, probability)
        self.effects.append(effect)

    def add_precond(self, name: str, objects: List[CustomObject]):
        """
        Creates a precondition and adds it to this action.

        Arguments
        ---------
        name : str
            The name of the predicate to be used for the precondition.
        objects : List
            The list of objects this predicate applies to.

        Returns
        -------
        None
        """
        for obj in objects:
            if obj not in self.obj_params:
                raise Exception(
                    "Object must be one of the objects supplied to the action"
                )
        precond = Fluent(name, objects)
        self.precond.append(precond)


class ObservationToken:
    """
    An ObservationToken object implements a `tokenize` function to generate an
    observation token for an action-state pair.
    """

    class token_method(Enum):
        IDENTITY = 1

    def __init__(
        self,
        method: Union[
            token_method, Callable[[Action, List[Fluent]], tuple]
        ] = token_method.IDENTITY,
    ):
        """
        Creates an ObservationToken object. This will store the supplied
        tokenize method to use on action-state pairs.

        Attributes
        ----------
        tokenize : function
            The function to apply to an action-state pair to generate the
            observation token.
        """
        if isinstance(method, self.token_method):
            self.tokenize = self.get_method(method)
        else:
            self.tokenize = method

    def get_method(
        self, method: token_method
    ) -> Callable[[Action, List[Fluent]], tuple]:
        """
        Retrieves a predefined `tokenize` function.

        Arguments
        ---------
        method : Enum
            The enum corresponding to the predefined `tokenize` function.

        Returns
        -------
        tokenize : object
            The `tokenize` function reference.
        """

        tokenize = self.identity
        if method == self.token_method.IDENTITY:
            tokenize = self.identity
        return tokenize

    def identity(self, action: Action, state: List[Fluent]) -> tuple:
        """
        The identity `tokenize` function.

        Arguments
        ---------
        action : Action
            An `Action` object.
        state : List
            An list of fluents representing the state.

        Returns
        -------
        tokenize : function
            The `tokenize` function reference.
        """
        return (action, state)


class State:
    """
    Class for a State, which is the set of all fluents and their values at a particular Step.

    Arguments
    ---------
    fluents : List of Fluents
            A list of fluents representing the state.
    """

    def __init__(self, fluents: List[Fluent]):
        self.fluents = fluents

    def __repr__(self):
        string = ""
        for fluent in self.fluents:
            string = string + fluent.name + ": " + str(fluent.value) + "\n"
        string = string.strip()
        return string

class Step():
    """
    A Step object stores the action, and state prior to the action for a step
    in a trace.
    """

    def __init__(self, action: Action, state: State):
        """
        Creates a Step object. This stores action, and state prior to the
        action.

        Attributes
        ----------
        action : Action
            The action taken in this step.
        state : State
            The state (list of fluents) at this step.
        """
        self.action = action
        self.state = state

class Trace:
    """
    Class for a Trace, which consists of each Step in a generated solution.

    Arguments
    ---------
    steps : List of Steps
        The list of Step objects that make up the trace.

    Other Class Attributes:
    num_fluents : int
        The number of fluents used.
    fluents : List of str
        The list of the names of all fluents used.
        Information on the values of fluents are found in the steps.
    actions: List of Actions
        The list of the names of all actions used.
        Information on the preconditions/effects of actions are found in the steps.

    """

    def __init__(self, steps: List[Step]):
        self.steps = steps
        self.num_fluents = len(steps)
        self.fluents = self.base_fluents()
        self.actions = self.base_actions()

    def base_fluents(self):
        """
        Retrieves the names of all fluents used in this trace.

        Returns
        -------
        list : str
            Returns a list of the names of all fluents used in this trace.
        """
        fluents = []
        for step in self.steps:
            for fluent in step.state.fluents:
                name = fluent.name
                if name not in fluents:
                    fluents.append(name)
        return fluents

    def base_actions(self):
        """
        Retrieves the names of all actions used in this trace.

        Returns
        -------
        list : str
            Returns a list of the names of all actions used in this trace.
        """
        actions = []
        for step in self.steps:
            name = step.action.name
            if name not in actions:
                actions.append(name)
        return actions

    def get_prev_states(self, action: Action):
        """
        Returns the state of the trace before this action.

        Arguments
        ---------
        action : Action
            An `Action` object.

        Returns
        -------
        prev_states : List of States
            A list of states before this action took place.
        """
        prev_states = []
        for step in self.steps:
            if step.action == action:
                prev_states.append(step.state)
        return prev_states

    def get_post_states(self, action: Action):
        """
        Returns the state of the trace after this Action.

        Arguments
        ---------
        action : Action
            An `Action` object.

        Returns
        -------
        post_states : List of States
            A list of states after this action took place.
        """
        post_states = []
        for i in range(len(self.steps) - 1):
            if self.steps[i].action == action:
                post_states.append(self.steps[i + 1].state)
        return post_states

class TraceList:
    """
    A TraceList object is a list-like object that holds information about a
    series of traces.
    """

    class MissingGenerator(Exception):
        def __init__(
            self,
            trace_list,
            message="TraceList is missing a generate function.",
        ):
            self.trace_list = trace_list
            self.message = message
            super().__init__(message)

    def __init__(self, generator: Union[Callable, None] = None):
        """
        Creates a TraceList object. This stores a list of traces and optionally
        the function used to generate the traces.

        Attributes
        ----------
        traces : List
            The list of `Trace` objects.
        generator : funtion | None
            (Optional) The function used to generate the traces.
        """
        self.traces: List[Trace] = []
        self.generator = generator

    def generate_more(self, num: int):
        if self.generator is None:
            raise self.MissingGenerator(self)

        self.traces.extend(self.generator(num))

    def get_usage(self, action: Action):
        usages = []
        for trace in self.traces:
            usages.append(trace.get_usage(action))
        return usages

    def __iter__(self):
        return TraceListIterator(self)


class TraceListIterator:
    def __init__(self, trace_list: TraceList):
        self._trace_list = trace_list
        self._index = 0

    def __next__(self):
        if self._index < len(self._trace_list.traces):
            item = self._trace_list.traces[self._index]
            self._index += 1
            return item
        raise StopIteration

    def __iter__(self):
        return self


if __name__ == "__main__":
    objects = [CustomObject(str(o)) for o in range(3)]
    action = Action("put down", objects)
    action2 = Action("pick up", objects)
    action3 = Action("restart", objects)
    # action.add_effect("eff", ["block 1", "block 2"], "func1", 94)
    # action.add_precond("precond", ["block 1", "block 2"])
    # action.add_effect("eff", ["block 1", "block 3"], "func1", 94)
    fluent = Fluent("on table", objects, True)
    fluent2 = Fluent("in hand", objects, True)
    fluent3 = Fluent("dropped", objects, False)

    s = Step(action, [fluent])

    o = ObservationToken()
    state = [fluent]
    token = o.tokenize(action, state)
    # print(token)

    state = State([fluent])
    state2 = State([fluent, fluent2])
    state3 = State([fluent, fluent2, fluent3])

    step = Step(action, state)
    step2 = Step(action2, state2)
    step3 = Step(action3, state3)
    
    trace = Trace([step, step2, step3])
    print(trace.steps)
    print(trace.num_fluents)
    print(trace.fluents)
    print(trace.actions)
    print(trace.get_prev_states(action2))
    print(trace.get_post_states(action2))

