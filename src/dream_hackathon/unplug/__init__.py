"""A scripted graph-as-policy that unplugs USB cables from a power strip with one Metal arm.

`dhk-collect --policy graph` runs it in place of Quest teleop (collect.py). The pieces, none of which opens a device:

- params.py: speeds, limits and vision thresholds (configs/unplug.yaml).
- poses.py: the arm's taught poses: a ready pose, each cable's grasp, the drop spot (configs/unplug_poses.yaml,
  written by scripts/mine_unplug_rollouts.py from recorded demonstrations, or by scripts/teach_unplug_poses.py).
- motion.py: straight-line tool paths solved to joints before the arm moves.
- verify.py: the wrist camera check that the plug's metal tip is out of its socket.
- graph.py: the state machine, one control tick at a time.
- mining.py: finds the grasps, lifts and drops in recorded episodes.
"""
