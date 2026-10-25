"""MuJoCo adapter for bipeds: settings, scene loading, and truth extraction.

Bipeds are the only robots simulated in MuJoCo; multirotors and wheeled
robots stay in Gazebo (docs/9_bipeds.md). MuJoCo is stepped explicitly
(``mj_step1``/``mj_step2``), so this is a step-based execution model,
kept separate from the asynchronous Gazebo transport. The
``mujoco`` package (the ``mujoco`` extra) is imported only inside the
functions that need it, so settings load and validate without it.
"""
