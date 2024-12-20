from pathlib import Path
from unittest import TestCase
from macq.trace import Fluent, PlanningObject, TraceList
from macq.generate.pddl import TraceFromGoal, VanillaSampling
from macq.extract import Model
from macq.extract.learned_fluent import PHashLearnedLiftedFluent
from macq.extract.esam import ESAM
from macq.observation.identity_observation import IdentityObservation


# !!!note to self: need to change goal fluents to the original goal(even if it set automatically)
# because an error accuses

def get_fluent(name: str, objs: list[str]):
    objects = [PlanningObject(o.split()[0], o.split()[1])
               for o in objs]
    return Fluent(name, objects)


class TestESAM(TestCase):
    # all logistic domain info for tests
    log_model: Model
    action_2_sort_log = {"load-truck": ["obj", "obj", "loc"],
                         "unload-truck": ["obj", "obj", "loc"],
                         "load-airplane": ["obj", "obj", "loc"],
                         "unload-airplane": ["obj", "obj", "loc"],
                         "drive-truck": ["obj", "loc", "loc", "cit"],
                         "fly-airplane": ["obj", "loc", "loc"]}

    def test_extraction_under_injuctive_assumption(self):
        generator: TraceFromGoal = TraceFromGoal(problem_id=1481, observe_pres_effs=True, observe_static_fluents=True)
        base = Path(__file__).parent.parent
        model_dom = str(
            (base / "pddl_testing_files/esam_pddl_files/new_domain.pddl").resolve()
        )
        model_prob = str(
            (base / "pddl_testing_files/esam_pddl_files/new_prob.pddl").resolve()
        )
        generator.change_goal({
            get_fluent(
                "at",
                ["package package6", "location city1-2"]
            ),
            get_fluent(
                "at",
                ["package package5", "location city6-2"]
            ),
            get_fluent(
                "at",
                ["package package4", "location city3-2"]
            ),
            get_fluent(
                "at",
                ["package package3", "location city6-1"]
            ),
            get_fluent(
                "at",
                ["package package2", "location city6-2"]
            ),
            get_fluent(
                "at",
                ["package package1", "location city2-1"]
            )
        }, model_dom, model_prob)
        traces = [generator.generate_trace()]

        # prob 2
        generator = TraceFromGoal(problem_id=1496, observe_pres_effs=True, observe_static_fluents=True)
        generator.change_goal({
            get_fluent(
                "at",
                ["package obj31", "location apt4"]
            ),
            get_fluent(
                "at",
                ["package obj22", "location apt2"]
            ),
            get_fluent(
                "at",
                ["package obj42", "location apt4"]
            ),
            get_fluent(
                "at",
                ["package obj53", "location apt3"]
            ),
            get_fluent(
                "at",
                ["package obj12", "location pos1"]
            ),
            get_fluent(
                "at",
                ["package obj32", "location apt1"]
            ),
            get_fluent(
                "at",
                ["package obj43", "location apt3"]
            ),
            get_fluent(
                "at",
                ["package obj52", "location apt1"]
            ),
            get_fluent(
                "at",
                ["package obj51", "location apt3"]
            ),
            get_fluent(
                "at",
                ["package obj21", "location apt4"]
            ),
            get_fluent(
                "at",
                ["package obj11", "location pos3"]
            ),
            get_fluent(
                "at",
                ["package obj23", "location pos4"]
            ),
            get_fluent(
                "at",
                ["package obj33", "location pos3"]
            ),
            get_fluent(
                "at",
                ["package obj13", "location apt3"]
            ),
            get_fluent(
                "at",
                ["package obj41", "location pos1"]
            )
        }, model_dom, model_prob)
        traces.append(generator.generate_trace())
        trace_list: TraceList = TraceList(traces=traces)
        from macq.observation import IdentityObservation
        esam_model: Model = ESAM(obs_trace_list=trace_list.tokenize(
            Token=IdentityObservation), debug=True)

        print(esam_model.details())
        print("\n\n\n\n===================================")
        model_dom = str(
            (base / "pddl_testing_files/esam_pddl_files/new_domain_copy.pddl").resolve()
        )
        model_prob = str(
            (base / "pddl_testing_files/esam_pddl_files/new_prob_copy.pddl").resolve()
        )
        esam_model.to_pddl('graph', 'log00_x', model_dom, model_prob)

    def test_extraction_random_sample(self):
        vanilla = VanillaSampling(problem_id=1481, observe_pres_effs=True, observe_static_fluents=True, plan_len=10)
        base = Path(__file__).parent.parent
        plan = vanilla.generate_plan()
        print(plan)
        print()
        vanilla.num_traces = 4
        vanilla.generate_traces()
        trace_list = vanilla.traces
        trace_list.generate_more(3)
        esam_model: Model = ESAM(obs_trace_list=trace_list.tokenize(
            Token=IdentityObservation), debug=True)

        print(esam_model.details())
        print("\n\n\n\n===================================")
        model_dom = str(
            (base / "pddl_testing_files/esam_pddl_files/new_domain_copy.pddl").resolve()
        )
        model_prob = str(
            (base / "pddl_testing_files/esam_pddl_files/new_prob_copy.pddl").resolve()
        )
        esam_model.to_pddl('graph', 'log00_x', model_dom, model_prob)

    def test_with_object_multiple_bindings(self):
        base = Path(__file__).parent.parent
        model_dom = str(
            (base / "pddl_testing_files/esam_pddl_files/graph_domain.pddl").resolve()
        )
        model_prob = str(
            (base / "pddl_testing_files/esam_pddl_files/graph_prob.pddl").resolve()
        )
        vanilla = VanillaSampling(dom=model_dom,
                                  prob=model_prob,
                                  observe_pres_effs=True,
                                  observe_static_fluents=True)

        plan = vanilla.generate_plan()
        print("the plan is:!!!")
        print(plan)
        print("end of plan=========!!!")

        trace_list: TraceList = TraceList()
        trace_list.append(vanilla.generate_single_trace_from_plan(plan=plan))
        esam_model: Model = ESAM(obs_trace_list=trace_list.tokenize(
            Token=IdentityObservation), debug=True)

        print(esam_model.details())
        print("\n\n\n\n===================================")
        model_dom = str(
            (base / "pddl_testing_files/esam_pddl_files/graph_domain_output.pddl").resolve()
        )
        model_prob = str(
            (base / "pddl_testing_files/esam_pddl_files/graph_prob_output.pddl").resolve()
        )
        esam_model.to_pddl_lifted('graph', 'log00_x', model_dom, model_prob)


    def test_with_object_multiple_bindings2(self):
        base = Path(__file__).parent.parent
        model_dom = str(
            (base / "pddl_testing_files/esam_pddl_files/graph_domain.pddl").resolve()
        )
        model_prob = str(
            (base / "pddl_testing_files/esam_pddl_files/graph_prob2.pddl").resolve()
        )
        vanilla = VanillaSampling(dom=model_dom,
                                  prob=model_prob,
                                  plan_len=2,
                                  num_traces=1,
                                  max_time=120,
                                  observe_static_fluents=True,
                                  observe_pres_effs=True)
        p = vanilla.generate_plan()
        trace_list: TraceList = TraceList([vanilla.generate_single_trace_from_plan(p)])

        trace_list.traces[0].print(view="actions")
        esam_model: Model = ESAM(obs_trace_list=trace_list.tokenize(
            Token=IdentityObservation), debug=True)

        print(esam_model.details())
        print("\n\n\n\n===================================")
        model_dom = str(
            (base / "pddl_testing_files/esam_pddl_files/graph_domain_output.pddl").resolve()
        )
        model_prob = str(
            (base / "pddl_testing_files/esam_pddl_files/graph_prob_output2.pddl").resolve()
        )
        esam_model.to_pddl_lifted('graph', 'log00_x', model_dom, model_prob)




    def test_minimize_param_act_inds(self):
        def test_minimize_param_act_inds1():
            flu1 = PHashLearnedLiftedFluent("a", ["t1"], param_act_inds=[0])
            flu2 = PHashLearnedLiftedFluent("a", ["t1"], param_act_inds=[1])
            model_dict: [PHashLearnedLiftedFluent, bool] = {flu1: True, flu2: True}
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=2)
            assert len(set(out.values())) == 1, "len of values should be 1"

            model_dict[flu2] = False
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=2)
            assert len(set(out.values())) == 2, "len of values should be 2"

        def test_minimize_param_act_inds2():
            flu1 = PHashLearnedLiftedFluent("a", ["t1", "t1"], param_act_inds=[0, 2])
            flu2 = PHashLearnedLiftedFluent("a", ["t1", "t1"], param_act_inds=[1, 2])
            flu3 = PHashLearnedLiftedFluent("a", ["t1", "t1"], param_act_inds=[0, 3])
            flu4 = PHashLearnedLiftedFluent("a", ["t1", "t1"], param_act_inds=[1, 3])
            model_dict: [PHashLearnedLiftedFluent, bool] = {flu1: True, flu2: True, flu3: True, flu4: True}
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=4)
            assert len(set(out.values())) == 2, "len of values should be 2"

            model_dict[flu4] = False
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=4)
            assert len(set(out.values())) == 4, "len of values should be 4"

            model_dict[flu3] = False
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=4)
            assert len(set(out.values())) == 4, "len of values should be 4"

            model_dict[flu2] = False
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=4)
            assert len(set(out.values())) == 4, "len of values should be 4"

        def test_minimize_param_act_inds3():
            flu1 = PHashLearnedLiftedFluent("a", ["t1"], param_act_inds=[0])
            flu2 = PHashLearnedLiftedFluent("a", ["t1"], param_act_inds=[1])
            flu3 = PHashLearnedLiftedFluent("b", ["t1"], param_act_inds=[2])
            flu4 = PHashLearnedLiftedFluent("b", ["t1"], param_act_inds=[3])
            model_dict: [PHashLearnedLiftedFluent, bool] = {flu1: True, flu2: True, flu3: True, flu4: True}
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=4)
            assert len(set(out.values())) == 2, "len of values should be 2"

            model_dict[flu4] = False
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=4)
            assert len(set(out.values())) == 3, "len of values should be 3"

            model_dict[flu3] = False
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=4)
            assert len(set(out.values())) == 3, "len of values should be 3"

            model_dict[flu2] = False
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=4)
            assert len(set(out.values())) == 4, "len of values should be 4"

        def test_minimize_param_act_inds4():
            flu1 = PHashLearnedLiftedFluent("a", ["t1"], param_act_inds=[0])
            flu2 = PHashLearnedLiftedFluent("a", ["t1"], param_act_inds=[1])
            flu3 = PHashLearnedLiftedFluent("a", ["t1"], param_act_inds=[2])
            flu4 = PHashLearnedLiftedFluent("a", ["t1"], param_act_inds=[3])
            model_dict: [PHashLearnedLiftedFluent, bool] = {flu1: True, flu2: True, flu3: True, flu4: True}
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=4)
            assert len(set(out.values())) == 1, "len of values should be 2"

            model_dict[flu4] = False
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=4)
            assert len(set(out.values())) == 2, "len of values should be 3"

            model_dict[flu3] = False
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=4)
            assert len(set(out.values())) == 3, "len of values should be 3"

            model_dict[flu2] = False
            out = ESAM.minimize_parameters(model_dict=model_dict, act_num_of_param=4)
            assert len(set(out.values())) == 4, "len of values should be 4"

        test_minimize_param_act_inds1()
        test_minimize_param_act_inds2()
        test_minimize_param_act_inds3()
        test_minimize_param_act_inds4()
