"""Ring-based pair assignment for distributed collision-pair detection.

Even N uses asymmetric hop counts (manuscript / benchmark eval). Odd N assigns
``fleet_size // 2`` hops to every robot, which partitions all unordered pairs
without duplication.
"""


def expected_pair_count(fleet_size):
    """Return the number of unordered agent pairs for a fleet of ``fleet_size``."""
    if fleet_size < 2:
        return 0
    return fleet_size * (fleet_size - 1) // 2


def _hop_count_for_fleet_index(fleet_index_1based, fleet_size):
    """Return how many ring hops this 1-based fleet index checks."""
    if fleet_size < 2:
        return 0
    if fleet_size % 2 == 0:
        half = fleet_size // 2
        return half if fleet_index_1based <= half else half - 1
    return fleet_size // 2


def assigned_pairs_for_fleet_index(fleet_index, fleet_size):
    """Return sorted agent-index pairs assigned to ``fleet_index`` (0-based).

    Pairs use 0-based agent indices consistent with ``MultiAgentSimultaneousPlanner``.
    """
    if fleet_size < 2:
        return []
    if fleet_index < 0 or fleet_index >= fleet_size:
        raise ValueError(f"fleet_index {fleet_index} out of range for fleet_size {fleet_size}")

    robot_id_1b = fleet_index + 1
    hop_count = _hop_count_for_fleet_index(robot_id_1b, fleet_size)
    pairs = []
    for hop in range(1, hop_count + 1):
        other_1b = ((robot_id_1b - 1 + hop) % fleet_size) + 1
        pair = tuple(sorted((robot_id_1b - 1, other_1b - 1)))
        pairs.append(pair)
    return pairs


def assigned_pairs_for_robot_id(robot_id, fleet_robot_ids):
    """Return sorted agent-index pairs assigned to ``robot_id`` in ``fleet_robot_ids``."""
    fleet = [int(x) for x in fleet_robot_ids]
    if robot_id not in fleet:
        raise ValueError(f"robot_id {robot_id} not in fleet_robot_ids {fleet_robot_ids}")
    fleet_index = fleet.index(int(robot_id))
    return assigned_pairs_for_fleet_index(fleet_index, len(fleet))


def expected_pair_partition(fleet_robot_ids):
    """Map each fleet robot id to its assigned (agent_index, agent_index) pairs."""
    fleet = [int(x) for x in fleet_robot_ids]
    return {
        fleet[idx]: assigned_pairs_for_fleet_index(idx, len(fleet))
        for idx in range(len(fleet))
    }


def validate_full_coverage(fleet_robot_ids):
    """Raise ``ValueError`` if assignment does not partition all unordered pairs once."""
    fleet = [int(x) for x in fleet_robot_ids]
    fleet_size = len(fleet)
    if fleet_size < 2:
        return

    partition = expected_pair_partition(fleet)
    seen = []
    for robot_id, pairs in partition.items():
        for pair in pairs:
            if pair[0] < 0 or pair[1] >= fleet_size:
                raise ValueError(f"Pair {pair} out of bounds for fleet_size {fleet_size}")
            if robot_id != fleet[pair[0]] and robot_id != fleet[pair[1]]:
                raise ValueError(
                    f"Robot {robot_id} assigned pair {pair} it is not part of"
                )
            seen.append(pair)

    if len(seen) != len(set(seen)):
        raise ValueError("Duplicate pair assignment detected")

    expected = expected_pair_count(fleet_size)
    if len(set(seen)) != expected:
        missing = []
        for a1 in range(fleet_size):
            for a2 in range(a1 + 1, fleet_size):
                if (a1, a2) not in set(seen):
                    missing.append((a1, a2))
        raise ValueError(
            f"Pair coverage mismatch: expected {expected}, got {len(set(seen))}, missing={missing[:5]}"
        )


def supports_distributed_assignment(fleet_robot_ids, allow_odd_fleet=False):
    """Return True when ring assignment is defined for this fleet."""
    fleet_size = len(fleet_robot_ids)
    if fleet_size < 2:
        return False
    if fleet_size % 2 == 0:
        return True
    return bool(allow_odd_fleet)
