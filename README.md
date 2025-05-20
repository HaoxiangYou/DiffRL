# D.VA
Official Implementation of Accelerating Visual-Policy Learning through Parallel Differentiable Simulation.

This codebase is built on top of [DiffRL](https://github.com/NVlabs/DiffRL) by Jie Xu, NIVIDA (Paper: https://arxiv.org/abs/2204.07137). 

We also acknowledge and credit various open-source implementations that helped us develop the baseline methods. These are listed at the end of this document.

## Installation

## Instruction

## Methods

## Results

## Citation

If you use this repo in your research, please consider citing the paper as follows:

```
@misc{you2025acceleratingvisualpolicylearningparallel,
      title={Accelerating Visual-Policy Learning through Parallel Differentiable Simulation}, 
      author={Haoxiang You and Yilang Liu and Ian Abraham},
      year={2025},
      eprint={2505.10646},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.10646}, 
}
```

## Licenses

All redistributed code from [DiffRL](https://github.com/NVlabs/DiffRL) retains its original [license](LICENSES/DiffRL/LICENSE.md).

XML files from [dm_control](https://github.com/google-deepmind/dm_control) are licensed under the [Apache 2.0 License](https://github.com/google-deepmind/dm_control/blob/main/LICENSE).

All other code in this repository is licensed under the MIT License.

## Acknowledgement

1. The main codebase is built on top of [DiffRL](https://github.com/NVlabs/DiffRL) by Jie Xu (NVIDIA).

2. CURL implementation is based on the original [repository](https://github.com/MishaLaskin/curl) by Michael Laskin.

3. DrQv2 implementation is based on the original [repository](https://github.com/facebookresearch/drqv2) by Denis Yarats (Facebook Research).

4. DreamerV3 implementation is based on the [pytorch implementation](https://github.com/NM512/dreamerv3-torch) by Naoki Morihira.

5. We refer the [pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) , developed by the Autonomous Robotic Manipulation Lab at the University of Michigan, Ann Arbor, to construct the forward kinematics tree used in our differentiable rendering pipeline.
We have made several modifications to support floating-base systems and multiple joints definition under single link.