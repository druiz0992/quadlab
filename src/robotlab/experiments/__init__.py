"""Experiment composition, execution, and saved result artifacts.

Entry point: ``run.run(GazeboConfig(...))`` picks the robot and its
simulator. Subpackages by role:

- ``harness``: what algorithm plug-ins plug into (loading, sensor-bus
  delivery, debug channel, metrics, offline replay, comparison, batches);
- ``common``: building blocks several robots share (world preparation, the
  Gazebo sensor pipeline, schedules, loop timing, interrupts);
- ``multirotor``, ``wheeled``, ``biped``: one mission composition and loop
  per robot family (Gazebo for drones and wheeled robots, MuJoCo for bipeds);
- ``calibration``: automated sensor calibration procedures.

Plug-in authors normally need only ``harness`` (and the configs); the rest
is orchestration.
"""
