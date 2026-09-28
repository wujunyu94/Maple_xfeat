"""Distance/velocity feedback for both navigation entry points.

Digital keys use predicted stopping distance instead of an integral term (which
would wind up against a platform edge). Far travel is a continuous hold.
"""


def walk_intervals(nav, intervals, timeout, precise, source, clock):
    if not intervals:
        return False
    end = clock.perf_counter()+timeout
    settled_at = braking_until = None
    previous = None
    corrections = 0
    nav.event('movement_begin', intervals=intervals, precise=precise,
              controller='distance_velocity_feedback_v2')

    def command(keys, reason, **data):
        nonlocal previous, corrections
        state = (keys, reason)
        nav.keys.set(*keys)
        if state != previous:
            if keys and previous is not None and not previous[0]:
                corrections += 1
            nav.event('movement_control', keys=list(keys), reason=reason, **data)
            previous = state

    while nav.alive() and clock.perf_counter() < end:
        now = clock.perf_counter()
        obs = nav.o.get()
        if source is not None and obs and obs['platform'] != source:
            command((), 'left_source_platform')
            return False
        # Full-resolution feature rescue takes ~100ms and emits every ~110ms.
        # A fixed 180ms capture-age limit would release for ~30ms every cycle.
        # Bridge only distant travel, bounded at 300ms (38.6px at speed cap).
        distance = min((max(lo-obs['world'][0], 0, obs['world'][0]-hi)
                        for lo, hi in intervals), default=0) if obs and obs.get('world') else 0
        max_age = .30 if distance > 60 else .18
        estimate = nav.motion.estimate(obs, now, max_age=max_age)
        if estimate is None:
            command((), 'missing_or_stale_ground_observation',
                    observation_age=now-obs['time'] if obs else None)
            settled_at = None
            clock.sleep(.01)
            continue
        raw, x, v = estimate['raw_x'], estimate['x'], estimate['vx']
        graph = getattr(nav, 'g', None)
        if graph is not None and now-obs['time'] > .18:
            node = graph.nodes.get(obs['platform'])
            if node is None or not node.x_min+8 <= estimate['stop_x'] <= node.x_max-8:
                command((), 'prediction_near_platform_edge')
                clock.sleep(.01)
                continue
        inside = any(lo <= raw <= hi for lo, hi in intervals)
        lo, hi = min(intervals, key=lambda r: max(r[0]-x, 0, x-r[1]))
        target = (lo+hi)/2 if precise else min(max(x, lo+min(12,(hi-lo)/2)), hi-min(12,(hi-lo)/2))
        error = target-x
        data = dict(raw_x=raw, filtered_x=x, vx=v, error=error,
                    stop_x=estimate['stop_x'], target_x=target,
                    observation_age=now-obs['time'], max_observation_age=max_age)
        if inside and abs(v) < 8:
            command((), 'settling', **data)
            settled_at = settled_at or now
            if now-settled_at >= .10:
                nav.event('movement_arrived', **data, corrections=corrections)
                return True
        else:
            settled_at = None
            sign = 1 if error > 0 else -1
            # Re-evaluate braking every frame; duration derives from speed,
            # never a fixed pulse/pause cadence. A short margin covers input lag.
            braking = braking_until is not None and now < braking_until
            if braking:
                command((), 'decelerating', **data)
            elif (inside and not precise) or (v*sign > 5 and sign*(target-estimate['stop_x']) <= 0):
                braking_until = now+abs(v)/nav.motion.model.drag_accel+.025
                command((), 'predicted_stop', brake_seconds=braking_until-now, **data)
            else:
                # At low speed a smaller error produces a smaller speed target;
                # stop-distance feedback releases earlier for the final correction.
                command(('right' if sign > 0 else 'left',),
                        'continuous_travel' if abs(error)>35 else 'near_target', **data)
        clock.sleep(.01)
    command((), 'timeout_or_stop')
    return False
