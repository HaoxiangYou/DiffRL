# D.VA
Official Implementation of 
<td style="padding:20px;width:75%;vertical-align:middle">
      <a href="https://haoxiangyou.github.io/Dva_website/" target="_blank">
      <b> Accelerating Visual-Policy Learning through Parallel Differentiable Simulation
      </b>
      </a>
      <br>
      <a href="https://haoxiangyou.github.io/" target="_blank">Haoxiang You</a>,
      <a href="https://yilangliu.github.io/" target="_blank">Yilang Liu</a> and
      <a href="https://ialab.yale.edu/" target="_blank">Ian Abraham</a>
      <br>
      <a href="https://www.arxiv.org/abs/2505.10646">paper</a> /
      <a href="https://haoxiangyou.github.io/Dva_website/" target="_blank">project page</a>
    <br>
</td>

<br>

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

<br>

This codebase is built on top of various open-source implementations, which we list at the end of this document.

## Installation


## Instruction

## Methods

## Results

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