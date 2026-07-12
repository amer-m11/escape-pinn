from enum import IntEnum


class NodeState(IntEnum):
    """The state of the node in the one-pass marching algorithm."""

    FAR = 0
    """No information yet. Not in the queue."""
    CONSIDERED = 1
    """Tentative values computed. In the priority queue."""
    ACCEPTED = 2
    """Final values computed. Will not change again."""
