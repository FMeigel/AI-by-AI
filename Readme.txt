In this folder, you find the the implementation of a code that developed in chat-conversation with ChatGPT.
When creating the code, I aimed for an didatic introduction and not so much for efficient implementation.
Keep this in mind when assessing the code for optimality.

This Code contains the following files:
1. Network.h is a header file containing the definition of the network and the neuron classes.
2. Train_NNCountRegion.cpp is the main file of the code. It builds a model data set, with the task to count the regions of consecutive 1s. It then builds, trains, and saves a neural network for this task.
3. Load_NNCountRegion.txt loads and tests the saved neural networks.
4. Networktestfile.txt is an example of such a saved network and can be loaded by Load_NNCountRegion.txt.
