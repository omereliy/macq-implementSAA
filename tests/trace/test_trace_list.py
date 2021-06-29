import pytest
from macq.trace import TraceList
from tests.utils.generators import (
    generate_test_trace_list,
    generate_test_trace,
    generate_blocks_traces,
)

MissingGenerator = TraceList.MissingGenerator


def test_trace_list():
    trace_list = generate_test_trace_list(5)

    assert len(trace_list) == 5

    with pytest.raises(MissingGenerator):
        trace_list.generate_more(5)

    first = trace_list[0]
    trace_list.generator = generate_test_trace
    trace_list.generate_more(5)
    assert len(trace_list) == 10
    assert trace_list[0] is first

    action = trace_list[0].steps[0].action
    usages = trace_list.get_usage(action)
    for i, trace in enumerate(trace_list):
        assert usages[i] == trace.get_usage(action)


if __name__ == "__main__":
    display = input("Display [details, color]: ")
    wrap = input("Wrap? [y, n]: ")
    traces = generate_blocks_traces()
    traces.print(display, wrap=(wrap.lower() == "y"))