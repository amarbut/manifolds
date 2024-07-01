#!/bin/bash

pip install git+https://github.com/huggingface/transformers

pip install torch transformers[torch] evaluate datasets scipy scikit-learn

pip install faiss-gpu faiss-cpu Isoscore
