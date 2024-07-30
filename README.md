# Apropos

## Spinning Up - Usage
TBD

## Spinning Up - Dev
#### 1. Pyenv
   /bash<br>
   `pyenv install 3.11.0`<br>
   `pyenv virtualenv 3.11.0 apropos-dev`<br>
   `pyenv activate apropos-dev`

#### 2. Poetry
   /bash<br>
   `curl -sSL https://install.python-poetry.org | python3 -`<br>
   `poetry install`


## Usage
If you use the MIPROv2 optimizer in your academic work, please cite the following [paper](https://arxiv.org/abs/2406.11695):

```
@misc{opsahlong2024optimizinginstructionsdemonstrationsmultistage,
      title={Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs}, 
      author={Krista Opsahl-Ong and Michael J Ryan and Josh Purtell and David Broman and Christopher Potts and Matei Zaharia and Omar Khattab},
      year={2024},
      eprint={2406.11695},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.11695}, 
}
```

Moreover, if you find the notion of learning from bootstrapped demonstrations useful, or have used algorithms such as the breadth-first random search optimizer, consider citing the following [paper](https://arxiv.org/abs/2310.03714)

```
@misc{khattab2023dspycompilingdeclarativelanguage,
      title={DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines}, 
      author={Omar Khattab and Arnav Singhvi and Paridhi Maheshwari and Zhiyuan Zhang and Keshav Santhanam and Sri Vardhamanan and Saiful Haq and Ashutosh Sharma and Thomas T. Joshi and Hanna Moazam and Heather Miller and Matei Zaharia and Christopher Potts},
      year={2023},
      eprint={2310.03714},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.03714}, 
}
```