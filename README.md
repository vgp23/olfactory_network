# Implementing biologically inspired olfactory classification in spiking neural networks 

This repository contains the implementation and example training + inference usage for the networks we trained. Full experiment details can be found in the accompanying report, but in summary we trained two types of spiking networks, in the same fashion as shown in [this paper](https://pubmed.ncbi.nlm.nih.gov/34619093/) on standard artificial networks. The control was trained with the [DECOLLE](https://arxiv.org/abs/1811.10766) algorithm (a local version of gradient descent), and the other network was trained using a variant of spike-timing-dependent plasticity in the last dimensionality-expanding layer, with DECOLLE everywhere else. Illustrations for these architectures can be found in the accompanying images folder.

## Installation

Use the provided .yml to build the required environment.

## Usage

Almost everything is shown in the `.ipynb` files. This creates all of the data we use, saves it, and also has you define the shape of the network. The current implementation is flexible: it can have as many layers with as many neurons as you wish. We worked with a much smaller dataset, defined on fewer classes than the original authors. You’ll find the original authors’ values in comments by the relevant parameters here.

STDP blocks can be placed wherever you wish, as long as it’s not the input layer (technical reasons, can be implemented but not so easily). This is defined using the `stdp_block_idx` parameter. One block consists of two layers (an input and an output, where one block’s output serves as the input for the next).

The network will automatically generate linear readout layers where required, such that it performs local backprop on all the blocks that are not the STDP block. It will also take care of a backprop-trainable linear readout layer at the end of the network for you, no need to define this in the `layer_sizes` list within the dataset creation section. As it’s currently configured, the network performs both batched backprop and batched STDP updates at each timepoint, and sets all the weights in the network to their absolute values except for the first block, as per the original paper.

The STDP rule itself is implemented in the `assistant.py` file, and there are 4 parameters to set (magnitude of weight updates for both LTP and LTD, and the decay constants for both). These are [not set in equal proportion to each other](https://www.frontiersin.org/articles/10.3389/fnins.2021.741116/full/); generally you potentiate more than you depress, but you have a longer time constant for depression. Not wanting to deal with 4 hyperparameters which were confusing, I instead collapsed them into two. The `stdp_learning_rate` controls both magnitude parameters and `stdp_tau_combined` controls both time constants. The idea is that as you change these, the pairs of parameters are updated in proportion to each other (I set a ratio of 2:5, arbitrarily). If you want to visualize what this does, you can directly do this within the assistant (I’ve shown this in the notebook as well)

## Important note

Watch out for the parameter `ignore_zeros` in `assistant.train_olfaction()`. When set to `True`, all instances where two neurons fire at exactly the same time are IGNORED. The reason is because our network isn’t very dynamic: the input spiking train rates are constant, once the input has been propagated to the end the network basically settles into a stable firing pattern. What this means is that basically every synapse that’s active in this pattern gets a ton of potentiation until we stop feeding input in. Ignoring these events allows the network to eventually stabilize. 