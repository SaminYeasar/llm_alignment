# llm_alignment
---
## DPO
* create env and install dependencies:
    ```
    conda create -n dpo python=3.9
    conda activate dpo
    pip install -r dpo/requirements.txt
    pip install --upgrade datasets
    ```

## Anole
* create env and install dependencies:
```
conda create -n anole python=3.11
conda activate anole
pip install -r anole/requirements.txt
pip install -e anole/transformers/.
```
* Anole dataset
    * details: https://github.com/SaminYeasar/llm_alignment/tree/main/facilitating_image_generation
* Train: 
    * follow the instruction on ReadMe from`https://github.com/SaminYeasar/llm_alignment/tree/main/anole/training` to train on custom-dataset