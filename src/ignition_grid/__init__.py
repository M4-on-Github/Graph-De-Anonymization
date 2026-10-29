"""E2 (Experiment Log 3): the Phase-3 ignition-threshold grid and the anchor audit.

    ignition_plan.py   which runs exist and how they are cut into array tasks
                       (standard library only; the head node runs it)
    ignition_grid.py   the runner: synthetic anchor set -> propagate -> record
    anchor_audit.py    step E2.0: the anchor sets the centrality_rank baseline used

Like src/adv_gd and src/baselines, these are run as scripts
(`python3 src/ignition_grid/ignition_grid.py ...`) and import their neighbours
by putting the neighbour's directory on sys.path, so nothing is imported here.
"""
