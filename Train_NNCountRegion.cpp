/*
This code trains a neural network to detect if the number of regions of consecutives 1s in sequence of 0 and 1 exceeds 5.
This task translates to the counting of horizontal lines in a BW-image.
The purpose of this code is to provide a simple, easy-to-understand implementation of a neural network.
While the task can be easily cast in logical operations, this projects serves as an illustrative proof of concept.
The code was developed in using ChatGTP [https://chat.openai.com] as an exercise to explore how well ChatGTP can assist with programming projects.
The structure of the classes and most style decisions were made by ChatGTP, with manual fixes for any inconsistencies.
*/

// Load libraries
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <string>

#include "Network.h"

using namespace std;


// Function to generate a binary vector of length 20 with random values of 0 or 1.
void generateBinary(vector<double>& binaryVec) {
    for (int i = 0; i < 20; i++) {
        binaryVec[i] = rand() % 2; // Generate a random value of 0 or 1 and store it in the binary vector.
    }
}



// Function to evaluate the output of the neural network given an input vector: Is the number of regions of consecutive 1's exceeding 5?
vector<double> evaluateOutput(vector<double> inputVector) {
    int numRegions = 0;
    bool inRegion = false;
    // Iterate through the input vector and count the number of regions of consecutive 1's.
    for (int i = 0; i < inputVector.size(); i++) {
        if (inputVector[i] == 1.0 && !inRegion) {  // Start of a new region.
            numRegions++;
            inRegion = true;
        } else if (inputVector[i] == 0.0) {  // End of a region.
            inRegion = false;
        }
    }

    // Create a vector with a single output value that is 1.0 if the number of regions is greater than 5, and 0.0 otherwise.
    vector<double> output(1, (numRegions > 5) ? 1.0 : 0.0);

    return output;
}


int main() {
    srand(time(NULL)); // Seed the random number generator with the current time

    // Create the training and testing data sets of size 100,000
    vector<vector<double> > trainInputs, trainOutputs, testInputs, testOutputs;
    for (int i = 0; i < 100000; i++) { // There are 2^20= 1,048,576 different combinations
        vector<double> binaryVec(20);
        generateBinary(binaryVec);
        vector<double> output;
        output= evaluateOutput(binaryVec); // Evaluate the output of the neural network for the binary vector
        
        // Split the generated data into training and testing data sets in a 95:5 ratio.
        if (i < 95000) {
            trainInputs.push_back(binaryVec);
            trainOutputs.push_back(output);
        }
        else {
            testInputs.push_back(binaryVec);
            testOutputs.push_back(output);
        }
    }


    // Create the neural network with 20 input neurons, a hidden layers, and 1 output neuron
    vector<int> topology;
    topology.push_back(20);
    topology.push_back(20);
    topology.push_back(10);
    topology.push_back(1);
    Network network(topology);


    // Train the neural network using backpropagation
    for (int i = 0; i < 100; i++) {

        // Shuffle the training data in each Epoch
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(trainInputs.begin(), trainInputs.end(), std::default_random_engine(seed));
        std::shuffle(trainOutputs.begin(), trainOutputs.end(), std::default_random_engine(seed));
        
        
        // Calculate the total error of the network over all training examples
        double error = 0.0;
        // Loop over all training examples
        for (int j = 0; j < trainInputs.size(); j++) {
            // Feed the input forward through the network
            network.feedForward(trainInputs[j]);

            // Apply backpropagation to update the weights
            network.backProp(trainOutputs[j]);

            // Add the average error of the network on this example to the total error
            error += network.getRecentAverageError();
        }

        cout << "Epoch " << i << ", error = " << error / trainInputs.size() << endl;
    }

    // Test the neural network on the test data set
    int correctCount = 0; // Initialize a counter for the number of correct predictions made by the network
    int randomGuess = 0; // Initialize a counter for the number of correct predictions made by random guessing
    double randomNumber = 0; // Initialize a variable for storing a random number
    int iterations = 100; // Define the number of iterations for testing the network. Alternative: int iterations = testInputs.size();


    for (int i = 0; i < iterations; i++) {

        network.feedForward(testInputs[i]); // Feed an input vector to the network
        double output = network.getOutputVals()[0]; // Get the output value of the network

        // Print the input vector
        cout << "Input: ";
        for (int j = 0; j < testInputs[i].size(); j++){
            cout<<testInputs[i][j];
        }
        cout << ", Value: " << testOutputs[i][0] << ", Prediction: " << output << endl;

        // Check if the prediction is correct and update the counter accordingly
        if ((output >= 0.5 && testOutputs[i][0] == 1.0) || (output < 0.5 && testOutputs[i][0] == 0.0)) {
            correctCount++;
        }

        // Generate a random prediction and check if it is correct and update the counter accordingly
        randomNumber= ((double) rand() / (RAND_MAX));
        if ((randomNumber >= 0.5 && testOutputs[i][0] == 1.0) || (randomNumber < 0.5 && testOutputs[i][0] == 0.0)) {
            randomGuess++;
        }
    }

    double accuracy = (double)correctCount / iterations; // Calculate the accuracy of the network
    double benchAccuracy = (double)randomGuess / iterations; // Calculate the accuracy of random guessing
    
    cout << "Accuracy = " << accuracy << ",  guessed Accuracy="<< benchAccuracy<< endl;

    //Save the network to a file
    string  filename = "./Networktestfile.txt";
    network.saveToFile(filename);

    return 0;
}
