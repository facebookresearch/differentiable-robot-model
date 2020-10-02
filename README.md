# differentiable robot model
Differentiable and learnable robot model. Our differentiable robot model implements computations such as 
forward kinematics and inverse dynamics, in a fully differentiable way. We also allow to specify  
parameters (kinematics or dynamics parameters), which can then be identified from data (see examples folder).

Currently, our code should work with any kinematic chain (eg any 7-DOF manipulator should work). It's been tested 
and evaluated particularly for the Kuka iiwa.


## Setup
```
conda create -n robot_model python=3.7
conda activate robot_model
python setup.py develop
```
Note that the data files might not be found if the setup is not run with `develop` (Fixme)

## Examples
2 examples scripts show the learning of kinematics parameters
```
python examples/learn_kinematics_of_iiwa.py
```

and the learning of dynamics parameters
```
python examples/learn_dynamics_of_iiwa.py
```

## L4DC paper and experiments
the notebook `experiments/l4dc-sim-experiments` shows a set of experiments that are similar to what we presented 
in our L4DC paper

```
@InProceedings{pmlr-v120-sutanto20a, 
    title = {Encoding Physical Constraints in Differentiable Newton-Euler Algorithm}, 
    author = {Sutanto, Giovanni and Wang, Austin and Lin, Yixin and Mukadam, Mustafa and Sukhatme, Gaurav and Rai, Akshara and Meier, Franziska}, 
    pages = {804--813}, 
    year = {2020},
    editor = {Alexandre M. Bayen and Ali Jadbabaie and George Pappas and Pablo A. Parrilo and Benjamin Recht and Claire Tomlin and Melanie Zeilinger}, 
    volume = {120}, 
    series = {Proceedings of Machine Learning Research}, 
    address = {The Cloud}, month = {10--11 Jun}, 
    publisher = {PMLR}, pdf = {http://proceedings.mlr.press/v120/sutanto20a/sutanto20a.pdf},
    url = {http://proceedings.mlr.press/v120/sutanto20a.html}, 
}
```

## Testing
running `pytest` in the top-level folder will run our differentiable robot model tests, 
which compare computations against pybullet.


## License

`differentiable-robot-model` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
See also our [Terms of Use](https://opensource.facebook.com/legal/terms) and [Privacy Policy](https://opensource.facebook.com/legal/privacy).
