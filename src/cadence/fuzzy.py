"""A Mamdani fuzzy inference system that sets green time from fuzzy inputs.

Traffic inputs have no crisp cutoffs: a queue of nine is "long" to some degree
and "medium" to another, and no threshold captures that honestly. A Mamdani FIS
fuzzifies each input into overlapping membership grades, fires a rule base with
min-implication, aggregates the rule outputs by max, and defuzzifies the
aggregate to a crisp green time by its centroid. Every step is written out - no
skfuzzy, no library - because the mechanism is the point.

The controller decides how long to hold the green for the phase about to run,
given the queue it will serve and the queue waiting on the other approach. When
its own queue is long and the cross queue short, it holds a long green; when its
queue is short, it yields early. A fixed 50/50 split cannot do that, and the
experiments measure the difference.
"""

from __future__ import annotations


def triangular(x: float, a: float, b: float, c: float) -> float:
    """Membership in a triangular set with feet at a, c and peak at b."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def shoulder_low(x: float, b: float, c: float) -> float:
    """A left shoulder: full membership up to b, sloping to zero at c."""
    if x <= b:
        return 1.0
    if x >= c:
        return 0.0
    return (c - x) / (c - b)


def shoulder_high(x: float, a: float, b: float) -> float:
    """A right shoulder: zero up to a, rising to full at b and staying there."""
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


class FuzzyGreenController:
    """Inputs: this phase's queue and the opposing queue (vehicles). Output: a
    green duration in seconds. The rule base is small and readable on purpose.

    Membership sets (queues, in vehicles):
        short  = low shoulder  [0 .. 8]
        medium = triangle      (3, 9, 15)
        long   = high shoulder [10 .. 20]
    Output green (seconds): short 10, medium 25, long 45 - singleton
    consequents, so the centroid defuzzification is a weighted average, which
    keeps the arithmetic transparent while remaining a true Mamdani aggregate.
    """

    GREEN = {"short": 10.0, "medium": 25.0, "long": 45.0}

    def _memberships(self, q: float) -> dict:
        return {
            "short": shoulder_low(q, 4.0, 10.0),
            "medium": triangular(q, 4.0, 10.0, 16.0),
            "long": shoulder_high(q, 10.0, 18.0),
        }

    def infer(self, my_queue: float, other_queue: float) -> float:
        mine = self._memberships(my_queue)
        other = self._memberships(other_queue)

        # Rule base. Each rule's firing strength is the min of its antecedents
        # (fuzzy AND), and it votes for a green length.
        rules = [
            (min(mine["short"], other["short"]), "short"),
            (min(mine["short"], other["medium"]), "short"),
            (min(mine["short"], other["long"]), "short"),
            (min(mine["medium"], other["short"]), "medium"),
            (min(mine["medium"], other["medium"]), "medium"),
            (min(mine["medium"], other["long"]), "short"),
            (min(mine["long"], other["short"]), "long"),
            (min(mine["long"], other["medium"]), "long"),
            (min(mine["long"], other["long"]), "medium"),
        ]

        # Aggregate by consequent (max), then defuzzify by centroid over the
        # singleton outputs: sum(strength * green) / sum(strength).
        strength = {"short": 0.0, "medium": 0.0, "long": 0.0}
        for w, label in rules:
            strength[label] = max(strength[label], w)

        num = sum(strength[k] * self.GREEN[k] for k in strength)
        den = sum(strength.values())
        if den == 0:
            return self.GREEN["medium"]      # no rule fired; a safe default
        return num / den

    def __call__(self, ns_queue: float, ew_queue: float, serving: int) -> float:
        if serving == 0:
            return self.infer(ns_queue, ew_queue)
        return self.infer(ew_queue, ns_queue)


class FixedGreenController:
    """The baseline: a constant green regardless of demand."""

    def __init__(self, green: float = 25.0):
        self.green = green

    def __call__(self, ns_queue: float, ew_queue: float, serving: int) -> float:
        return self.green
