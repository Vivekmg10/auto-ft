from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from autofinetune.graph.state import ExperimentState
from autofinetune.graph.nodes import (
    init_node,
    data_prep_node,
    planning_node,
    training_node,
    eval_node,
    update_node,
    report_node,
    error_node,
)
from autofinetune.graph.edges import should_continue, skip_eval_if_failed
from autofinetune.config.loader import ExperimentConfig


def build_graph(
    config: ExperimentConfig,
    strategist,
    data_agent,
    monitor,
    evaluator,
    memory,
    journal,
    storage,
):
    """
    Builds and compiles the experiment graph.
    All agents and services are injected here — nodes are pure functions.
    """

    # bind dependencies to each node via partial
    # this keeps nodes as plain functions while injecting what they need
    nodes = {
        "init": partial(init_node,
            config=config,
            storage=storage
        ),
        "data_prep": partial(data_prep_node,
            config=config,
            data_agent=data_agent
        ),
        "planning": partial(planning_node,
            config=config,
            strategist=strategist,
            memory=memory
        ),
        "training": partial(training_node,
            config=config,
            monitor=monitor,
            storage=storage
        ),
        "eval": partial(eval_node,
            config=config,
            evaluator=evaluator,
            storage=storage
        ),
        "update": partial(update_node,
            config=config,
            memory=memory,
            journal=journal,
            storage=storage
        ),
        "report": partial(report_node,
            journal=journal,
            storage=storage
        ),
        "error": partial(error_node,
            storage=storage
        ),
    }

    # bind config to edges
    _should_continue = partial(should_continue, config=config)

    # build the graph
    graph = StateGraph(ExperimentState)

    # add all nodes
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    # linear edges
    graph.set_entry_point("init")
    graph.add_edge("init", "data_prep")
    graph.add_edge("data_prep", "planning")
    graph.add_edge("planning", "training")

    # conditional after training — skip eval if failed
    graph.add_conditional_edges(
        "training",
        skip_eval_if_failed,
        {
            "eval": "eval",
            "update": "update",
        }
    )

    graph.add_edge("eval", "update")

    # main loop decision — continue or stop
    graph.add_conditional_edges(
        "update",
        _should_continue,
        {
            "planning": "planning",
            "report": "report",
            "error": "error",
        }
    )

    # terminal nodes
    graph.add_edge("report", END)
    graph.add_edge("error", END)

    # compile with in-memory checkpointer
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled


def create_initial_state(config: ExperimentConfig) -> ExperimentState:
    """
    Creates the initial state for a new experiment.
    If resuming, load from state.json instead.
    """
    return ExperimentState(
        experiment_id=config.id,
        use_case=config.experiment.use_case if hasattr(config, 'experiment') else config.use_case,
        max_runs=config.training.max_runs,
    )