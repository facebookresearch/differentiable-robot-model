# differentiable robot model

[![CircleCI](https://circleci.com/gh/facebookresearch/differentiable-robot-model/tree/main.svg?style=shield&circle-token=9bfa34219fadf44bb2b800d9a9bad3e00815fedf)](https://circleci.com/gh/facebookresearch/differentiable-robot-model/tree/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Differentiable and learnable robot model. Our differentiable robot model implements computations such as 
forward kinematics and inverse dynamics, in a fully differentiable way. We also allow to specify  
parameters (kinematics or dynamics parameters), which can then be identified from data (see examples folder).

Currently, our code should work with any kinematic trees. This package comes with wrappers specifically for:
* TriFinger Edu
* Kuka iiwa
* Franka Panda
* Allegro Hand
* Fetch Arm
* a 2-link toy robot

You can find the documentation here:  [Differentiable-Robot-Model Documentation](https://fmeier.github.io/differentiable-robot-model-docs/)  

## Installation
Requirements: python>= 3.7  

clone this repo and install from source:
```
git clone git@github.com:facebookresearch/differentiable-robot-model.git
cd differentiable-robot-model
python setup.py develop
```

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

## Code Contribution

We enforce linters for our code. The `formatting` test will not pass if your code does not conform.

To make this easy for yourself, you can either
- Add the formattings to your IDE
- Install the git [pre-commit](https://pre-commit.com/) hooks by running
    ```bash
    pip install pre-commit
    pre-commit install
    ```

For Python code, use [black](https://github.com/psf/black).

To enforce this in VSCode, install [black](https://github.com/psf/black), [set your Python formatter to black](https://code.visualstudio.com/docs/python/editing#_formatting) and [set Format On Save to true](https://code.visualstudio.com/updates/v1_6#_format-on-save).

To format manually, run: `black .`

## License

`differentiable-robot-model` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
See also our [Terms of Use](https://opensource.facebook.com/legal/terms) and [Privacy Policy](https://opensource.facebook.com/legal/privacy).
