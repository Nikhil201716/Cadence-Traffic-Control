"""A store-and-forward traffic simulator - the objective function every
controller in this project is scored against.

Vehicles are modelled as queues rather than individual agents: this is the
standard queueing (store-and-forward) model for signal-timing studies, it is
deterministic given a seed, and delay has a clean definition. By Little's law
the total delay over a run is the integral of queue length over time, so summing
every approach's queue at every second gives total vehicle-seconds of delay -
the single number to minimise.

Two contexts share the model:

  * an isolated intersection, where the question is how to split green time
    between competing approaches (the fuzzy controller's job); and
  * an arterial of several intersections, where the question is how to offset
    their green windows so a platoon rides a "green wave" (the genetic
    algorithm's job).
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class ArterialResult:
    total_delay: float          # vehicle-seconds, all approaches
    arterial_delay: float       # vehicle-seconds on the main street only
    cross_delay: float
    arterial_stops: int         # platoon arrivals that met a red
    vehicles_served: float

    def as_dict(self) -> dict:
        return {"total_delay": round(self.total_delay, 1),
                "arterial_delay": round(self.arterial_delay, 1),
                "cross_delay": round(self.cross_delay, 1),
                "arterial_stops": self.arterial_stops,
                "vehicles_served": round(self.vehicles_served, 1)}


def simulate_arterial(offsets: list[int], cycle: int, arterial_green: int,
                      travel_time: int, inflow: float, cross_inflow: float,
                      sat_flow: float, horizon: int,
                      lost_time: int = 2) -> ArterialResult:
    """Simulate a one-way arterial of ``len(offsets)`` signalised intersections.

    Signal i shows the arterial a green window of ``arterial_green`` seconds
    starting at ``offsets[i]`` within each ``cycle``; the rest of the cycle
    (minus ``lost_time`` for the amber/all-red) serves the cross street.
    Arterial arrivals at intersection i are intersection (i-1)'s departures
    delayed by ``travel_time``; the first intersection is fed by ``inflow``.
    Cross arrivals are an independent constant demand.

    The only thing the offsets change is *when* each green window falls, so a
    good offset vector lets a platoon released upstream arrive during the next
    signal's green and pass without stopping - and a bad one makes it stop at
    every signal.
    """
    n = len(offsets)
    arterial_q = [0.0] * n
    cross_q = [0.0] * n
    # departures[i][t] = vehicles that left intersection i's arterial approach at t
    departures = [[0.0] * (horizon + travel_time + 1) for _ in range(n)]

    total_delay = arterial_delay = cross_delay = 0.0
    stops = 0
    served = 0.0

    def arterial_is_green(i: int, t: int) -> bool:
        phase = (t - offsets[i]) % cycle
        return phase < arterial_green

    for t in range(horizon):
        for i in range(n):
            # -- arterial arrivals this second
            if i == 0:
                arr = inflow
            else:
                src = t - travel_time
                arr = departures[i - 1][src] if src >= 0 else 0.0
            arterial_q[i] += arr

            # -- cross arrivals
            cross_q[i] += cross_inflow

            green = arterial_is_green(i, t)
            # lost time at the very start of each green: no discharge yet
            phase = (t - offsets[i]) % cycle
            usable = green and phase >= lost_time

            if usable:
                depart = min(arterial_q[i], sat_flow)
                arterial_q[i] -= depart
                departures[i][t] = depart
                served += depart if i == n - 1 else 0.0
                # a platoon that arrived to a red earlier is a stop; count a
                # stop whenever fresh arrivals could not immediately pass
                if arr > 0 and arterial_q[i] > arr:
                    stops += 1
            else:
                if arr > 0:
                    stops += 1
                # cross street discharges when the arterial is red (minus lost)
                cross_phase = phase - arterial_green
                if not green and cross_phase >= lost_time:
                    cross_q[i] -= min(cross_q[i], sat_flow)

            arterial_delay += arterial_q[i]
            cross_delay += cross_q[i]

        total_delay = arterial_delay + cross_delay

    return ArterialResult(total_delay, arterial_delay, cross_delay, stops, served)


@dataclass
class IsolatedResult:
    total_delay: float
    avg_delay: float
    max_queue: float
    vehicles_served: float

    def as_dict(self) -> dict:
        return {"total_delay": round(self.total_delay, 1),
                "avg_delay": round(self.avg_delay, 3),
                "max_queue": round(self.max_queue, 1),
                "vehicles_served": round(self.vehicles_served, 1)}


def simulate_isolated(controller, demand: list[tuple[float, float]],
                      sat_flow: float, seconds_per_step: int = 1,
                      min_green: int = 5, max_green: int = 60,
                      lost_time: int = 4, seed: int = 0) -> IsolatedResult:
    """One intersection, two competing phases (say NS and EW), a controller that
    decides how long to hold each green.

    ``demand`` is a list of (ns_rate, ew_rate) in vehicles/second, one entry per
    control interval, so the demand can be made asymmetric and time-varying -
    which is exactly where a fixed 50/50 split wastes green on an empty approach
    and an adaptive controller earns its keep.

    ``controller`` is called as ``controller(ns_queue, ew_queue, serving)`` and
    returns the green duration (seconds) for the phase about to run; it is
    clamped to ``[min_green, max_green]``.
    """
    rng = random.Random(seed)
    ns_q = ew_q = 0.0
    total_delay = 0.0
    served = 0.0
    max_queue = 0.0
    t = 0
    serving = 0            # 0 = NS has green, 1 = EW
    interval = 0

    def rates() -> tuple[float, float]:
        return demand[min(interval, len(demand) - 1)]

    total_seconds = len(demand) * max_green
    while t < total_seconds:
        ns_rate, ew_rate = rates()
        green = controller(ns_q, ew_q, serving)
        green = max(min_green, min(max_green, int(green)))

        for step in range(green + lost_time):
            # Poisson-ish arrivals: a fractional rate as a Bernoulli per second
            ns_q += 1 if rng.random() < ns_rate else 0
            ew_q += 1 if rng.random() < ew_rate else 0
            # The startup lost time at the head of each green discharges nothing
            # - drivers reacting, vehicles accelerating - which is the real cost
            # that makes very short greens inefficient.
            if step >= lost_time:
                if serving == 0:
                    d = min(ns_q, sat_flow)
                    ns_q -= d
                    served += d
                else:
                    d = min(ew_q, sat_flow)
                    ew_q -= d
                    served += d
            total_delay += ns_q + ew_q
            max_queue = max(max_queue, ns_q, ew_q)
            t += 1
        serving ^= 1
        interval += 1

    return IsolatedResult(total_delay, total_delay / max(served, 1.0),
                          max_queue, served)
