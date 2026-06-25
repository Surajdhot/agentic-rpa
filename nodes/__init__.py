"""LangGraph node implementations for Conductor.

Each node is a small async function operating on :class:`ConductorState`.
The conditional router (:func:`check_node`) lives here too but is wired as an
edge function, not as a state-mutating node.
"""
