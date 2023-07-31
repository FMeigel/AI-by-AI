/*
This header file defines the "Network", "Neuron", and "Connection" classes for training and loading a deep neural network.
The purpose of this code is to provide a simple, easy-to-understand implementation of a neural network.
The code was developed in using ChatGTP [https://chat.openai.com] as an exercise to explore how well ChatGTP can assist with programming projects.
The structure of the classes and most style decisions were made by ChatGTP, with manual fixes for any inconsistencies.
*/


// Load libraries
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <ctime>

using namespace std;

// A class representing a single connection between neurons in a neural network
class Connection {
public:
    double weight;      // The weight associated with this connection
    double deltaWeight; // The change in weight associated with this connection (used in backpropagation)
    Connection();       // Default constructor
};

typedef vector<class Neuron> Layer;

// A class representing a single neuron in a neural network
class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned myIndex);            // Constructor that takes the number of outputs this neuron will have and its index in the current layer
    void feedForward(const Layer &prevLayer);                 // Feeds forward the output value of this neuron to the next layer
    void calcOutputGradients(double targetVal);               // Calculates the output gradients for this neuron, given a target output value
    void calcHiddenGradients(const Layer &nextLayer);         // Calculates the hidden gradients for this neuron, given the next layer's gradients
    void updateInputWeights(Layer &prevLayer);                // Updates the input weights of this neuron, given the previous layer

    void setOutputVal(double val) { m_outputVal = val; }      // Setter for the output value of the neuron
    double getOutputVal(void) const { return m_outputVal; }   // Getter for the output value of the neuron
         
    void setBias(double bias) { m_biasWeight = bias; }   // Setter for the bias weight
    double getBias(void) const { return m_biasWeight; }  // Getter for the bias weight

    void setOutputWeight(unsigned index, double weight) { m_outputWeights[index].weight = weight; } // Setter for the weight of the output connection at the given index
    double getOutputWeight(unsigned index) const { return m_outputWeights[index].weight; }          // Getter for the weight of the output connection at the given index
    

private:
    private:
    // Static variables for the overall net training rate, momentum, and transfer function
    static double eta;      // [0.0..1.0] overall net training rate
    static double alpha;    // [0.0..n] multiplier of last weight change (momentum)

    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);

    // Private functions for generating random weights and calculating the sum of deltas in the next layer
    static double randomWeight();
    double sumDOW(const Layer &nextLayer) const;

        
    // Private member variables for the output value, output weights, index in the current layer, and gradient
    double m_outputVal;                 // The output value of the neuron
    vector<Connection> m_outputWeights; // The output weights of the neuron
    unsigned m_myIndex;                 // The index of the neuron in its layer
    double m_gradient;                  // The gradient of the neuron, used for backpropagation


    // Additional private member variables for the bias neuron
    double m_biasWeight;         // The bias for each neuron
    double m_biasWeightDelta;    // Delta weight for the bias 
    double m_biasOutputVal;      // Output value for the bias neuron
};

double Neuron::eta = 0.01;     // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.00;   // momentum, multiplier of last deltaWeight, [0.0..1.0]


// A class representing a neural network
class Network {
public:
    Network(const vector<int> &topology);               // Constructor that initializes the neural network with the specified topology
    void initialize(const vector<unsigned>& topology);  // Resets and initializes and the neural network with the specified topology
    void feedForward(const vector<double> &inputVals);  // Feeds the input values forward through the network and calculates the output
    void backProp(const vector<double> &targetVals);    // Performs backpropagation with the specified target values
    void getResults(vector<double> &resultVals) const;  // Gets the output values of the neural network
   
    void saveToFile(const string& filename) const;      // Saves the neural network to a file with the specified filename
    bool loadFromFile(const string& filename);          // Loads the neural network from a file with the specified filename

    vector<double> getOutputVals() const {return m_outputVals; }               // Returns the output values of the neural network
    double getRecentAverageError(void) const {return m_recentAverageError; }   // Returns the recent average error of the neural network

private:
    
    vector<Layer> m_layers;                         // The layers of the neural network. Structure: m_layers[layerNum][neuronNum]
    double m_error;                                 // The current error of the neural network
    double m_recentAverageError;                    // The recent average error of the neural network
    static double m_recentAverageSmoothingFactor;   // The smoothing factor used for calculating the recent average error
    vector<double> m_outputVals;                    // The output values of the neural network after a feedforward pass
};


double Network::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over



// ******************************************************
// ****************** class Connection ******************
// ******************************************************

// Default constructor
Connection::Connection() {
    weight = 0.0;
    deltaWeight = 0.0;
}


// ******************************************************
// ****************** class Neuron **********************
// ******************************************************

// Constructor that takes the number of outputs this neuron will have and its index in the current layer
Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {

    // Initialize the output weights vector with numOutputs number of Connection objects
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex; // Set the index of the neuron in its layer
    m_outputVal = 0.0;   // Initialize the output value to 0
    m_gradient = 0.0;    // Initialize the gradient to 0


    m_biasWeight = randomWeight();  // Initialize bias weight to a random value between 0 and 1
    m_biasOutputVal = 1.0;          // Set bias output value to 1
    m_biasWeightDelta = 0.0;        // Set initial bias weight delta to 0.0

}

// Feeds forward the output value of this neuron to the next layer
void Neuron::feedForward(const Layer& prevLayer) {
    double sum = 0.0;
    // Sum the outputs from the previous layer's neurons
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    // Add the bias
    sum += m_biasOutputVal * m_biasWeight;

    // Apply the transfer function to calculate the output value
    m_outputVal = transferFunction(sum);
}

// Calculates the output gradients for this neuron, given a target output value
void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta * transferFunctionDerivative(m_outputVal);
}

// Calculates the hidden gradients for this neuron, given the next layer's gradients
void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * transferFunctionDerivative(m_outputVal);
}

// Updates the input weights of this neuron, given the previous layer
void Neuron::updateInputWeights(Layer &prevLayer) {
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for (unsigned n = 0; n < prevLayer.size(); ++n) {

        // Get reference to current neuron in previous layer
        Neuron &neuron = prevLayer[n];

        // Calculate delta weight using backpropagation algorithm
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;

        // Update delta weight and weight for the connection between the current neuron and the neuron in the previous layer
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;  
    }

    // Calculate bias weight
    double oldBiasDeltaWeight = m_biasWeightDelta;
    double newBiasDeltaWeight = eta * 1.0  * m_gradient + alpha * oldBiasDeltaWeight;

    // Update bias weight
    m_biasWeightDelta = newBiasDeltaWeight;
    m_biasWeight += newBiasDeltaWeight;
}


// Leaky ReLU as transfer function
double Neuron::transferFunction(double x) {
    if (x > 0) {
        return x;
    } else {
        return 0.1 * x;
    }
}

double Neuron::transferFunctionDerivative(double x) {
    if (x > 0) {
        return 1;
    } else {
        return 0.1;
    }
}

// Sigmoid as transfer function
/*double Neuron::transferFunction(double x) {
    return 1 / (1 + exp(-x));
}

double Neuron::transferFunctionDerivative(double x) {
    double fx = transferFunction(x);
    return fx * (1 - fx);
}*/


// This function generates a random weight for the connection between neurons.
double Neuron::randomWeight() {
    static default_random_engine generator;
    static normal_distribution<double> distribution(0.0, 0.5);
    return distribution(generator);
}

// Calculates the sum of the products of weights and gradients of the next layer neurons that are connected to the current neuron
double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;
    
    // Sum the contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < nextLayer.size(); ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}


// ******************************************************
// ****************** class Network *********************
// ******************************************************

// Constructor that initializes the neural network with the specified topology
Network::Network(const vector<int> &topology) {
    // Get the number of layers
	unsigned numLayers = topology.size();

    // Add layers
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// We have a new layer, now fill it with neurons
		for (unsigned neuronNum = 0; neuronNum < topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "Made a Neuron!" << endl;
		}

	}
}

// Resets and initializes and the neural network with the specified topology
void Network::initialize(const vector<unsigned>& topology) {   
    // Get the number of layers
    unsigned numLayers = topology.size();

    // Reset the current network
    m_layers.clear();

    // Add layers
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// We have a new layer, now fill it with neurons
		for (unsigned neuronNum = 0; neuronNum < topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "Made a Neuron!" << endl;
		}

	}
}

// Feeds the input values forward through the network and calculates the output
void Network::feedForward(const vector<double> &inputVals) {
    //cout << inputVals.size()<< ","<< m_layers[0].size() - 1 <<endl;
	assert(inputVals.size() == m_layers[0].size());

	// Assign the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// forward propagate
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size(); ++n) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}

    // Copy output neurons' values to m_outputVals
    m_outputVals.clear();
    const Layer& outputLayer = m_layers.back();
    for (unsigned n = 0; n < outputLayer.size(); ++n) {
        m_outputVals.push_back(outputLayer[n].getOutputVal());
    }
}


// Performs backpropagation with the specified target values
void Network::backProp(const vector<double> &targetVals) {

	// Calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = m_layers.back();
	m_error = 0.0;

    // Loop through each neuron in the output layer and calculate the error between the target value and the actual output value.
	for (unsigned n = 0; n < outputLayer.size() ; ++n) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta; // Add the squared error to the overall net error.
	}

	m_error /= outputLayer.size(); // Get average error squared
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)/ (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients
    for (unsigned n = 0; n < outputLayer.size(); ++n) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calculate gradients on hidden layers
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size(); ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}


// Gets the output values of the neural network
void Network::getResults(vector<double> &resultVals) const {
	resultVals.clear();

    // Loop through the neurons in the output layer and push their output values to the resultVals vector
    for (unsigned n = 0; n < m_layers.back().size(); ++n) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

// Saves the neural network to a file with the specified filename
void Network::saveToFile(const string& filename) const {
    ofstream file(filename);
    if (file.is_open()) {

        // Write the network topology to the file
        file << "Topology: ";
        for (unsigned i = 0; i < m_layers.size(); ++i) {
            file << m_layers[i].size() << " ";
        }
        file << endl;

        file << endl;

        //Write the weights to the file
        for (unsigned i = 0; i < m_layers.size() - 1; ++i) {
            const Layer& currentLayer = m_layers[i];

            // Iterate through each neuron in the current layer
            for (unsigned n = 0; n < currentLayer.size(); ++n) {
                const Neuron& neuron = currentLayer[n];
                
                // Iterate through each neuron in the next layer to get the weights
                for (unsigned neuronIndex = 0; neuronIndex < m_layers[i+1].size(); ++neuronIndex) {
                    
                    file << neuron.getOutputWeight(neuronIndex) << " ";
                    
                }
                file << endl;
            }
        }

        file << endl;

        // Write the bias weights to the file
        for (unsigned i = 1; i < m_layers.size(); ++i) {
            const Layer& currentLayer = m_layers[i];

            // Iterate through each neuron in the current layer
            for (unsigned n = 0; n < currentLayer.size(); ++n) {
                const Neuron& neuron = currentLayer[n];

                file << neuron.getBias() << endl;
            }
        }

        file.close();
        cout << "Network saved to file: " << filename << endl;
    } else {
        cerr << "Error: Unable to open file " << filename << " for writing." << endl;
    }
}

// Loads the neural network from a file with the specified filename
bool Network::loadFromFile(const string& filename) {
    ifstream file(filename);
    if (file.is_open()) {
        string line;
        vector<unsigned> topology;
        unsigned numLayers = 0;

        // Read the topology from the file
        getline(file, line);
        stringstream ss(line);
        string tag;
        ss >> tag;
        assert(tag == "Topology:");
        while (ss >> tag) {
            topology.push_back(stoi(tag));
        }
        numLayers = topology.size();

        // Initialize the network with the topology
        initialize(topology);
        

        // Read the weights from the file
        for (unsigned i = 0; i < numLayers - 1; ++i) {
            Layer& currentLayer = m_layers[i];

            // Iterate through each neuron in the current layer
            for (unsigned n = 0; n < currentLayer.size(); ++n) {
                Neuron& neuron = currentLayer[n];

                // Iterate through each neuron in the next layer to set the weights
                for (unsigned neuronIndex = 0; neuronIndex < m_layers[i+1].size(); ++neuronIndex) {

                    //Set the weights
                    double weight;
                    file >> weight;
                    neuron.setOutputWeight(neuronIndex, weight);
                }
            }
        }

        // Read the bias weights from the file
        for (unsigned i = 1; i < numLayers; ++i) {
            Layer& currentLayer = m_layers[i];

            // Iterate through each neuron in the current layer
            for (unsigned n = 0; n < currentLayer.size(); ++n) {
                Neuron& neuron = currentLayer[n];

                //Set the bias
                double bias;
                file >> bias;
                neuron.setBias(bias);
            }
        }

        file.close();

        cout << "Network loaded from file: " << filename << endl;
        return true;
    } else {
        cerr << "Error: Unable to open file " << filename << " for reading." << endl;
        return false;
    }
}







