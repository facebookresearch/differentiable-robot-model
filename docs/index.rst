Differentiable Robot Model Documentation
========================================

Overview
-------------

Our differentiable robot model library implements computations such as
forward kinematics and inverse dynamics, in a fully differentiable way. We also allow to specify
parameters (kinematics or dynamics parameters), which can then be identified from data (see examples folder).

Currently, our code should work with any kinematic trees. This package comes with wrappers specifically for:

   * Kuka iiwa
   * Franka Panda
   * Allegro Hand
   * a 2-link toy robot

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: API:

   modules/index




Indices and tables
==================

* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`
