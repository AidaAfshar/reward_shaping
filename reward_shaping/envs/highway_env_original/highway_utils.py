def longitudinal_distance(v1, v2, c=None):
    """
    The longitudinal distance between two cars v1 and v2
    v1 is intended to be BEHIND v2
    """
    # First: Check if v2 is geometrically in front of v1 or not
    if not behind(v1, v2, c):
        return float('inf')  # an arbitrary large value.
    else:
        dist = longitudinal_road_distance(v1, v2, c)
        dist -= c["vehicle_length"]  # (ego_length + actor_length) approximated
        return max(dist, 0)


def lateral_distance(v1, v2, c=None):
    if same_lane(v1, v2, c):
        return 0.0
    if left_lane(v1, v2, c):
        left_car = v1
        right_car = v2
    else:
        left_car = v2
        right_car = v1
    dist = lateral_road_distance(left_car, right_car, c)
    dist -= c["vehicle_width"]  # (ego_width + actor_width) approximated
    return max(dist, 0)


def lateral_road_distance(v1, v2, c=None):
    return v2[2] - v1[2]  # v[2] ~ y


def longitudinal_road_distance(v1, v2, c=None):
    return v2[1] - v1[1]  # v[1] ~ x


def behind(v1, v2, c=None):
    return True if v2[1] > v1[1] else False  # v[1] ~ x


def same_lane(v1, v2, c=None):
    return True if lane_id(v1[2], c) == lane_id(v2[2], c) else False  # v[2] ~ y


def left_lane(v1, v2, c=None):
    return True if lane_id(v1[2], c) < lane_id(v2[2], c) else False


def lane_id(y, c=None):
    return int(y / c["lane_width"])


def in_vicinity(v1, v2, c=None, vicinity=60):
    if (longitudinal_distance(v1, v2, c) < vicinity) or \
            (longitudinal_distance(v2, v1, c) < vicinity):
        return True
    else:
        return False
