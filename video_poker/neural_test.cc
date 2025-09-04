#include "neural.h"
#include "activations.h"

#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

// Test the feed-forward functionality of the NeuralNet.
void test_feed_forward() {
    std::vector<LayerSpecification> topology = { 
        {5, sigmoid, sigmoid}, 
        {10, sigmoid, sigmoid}, 
        {2, sigmoid, sigmoid}, 
    };
    NeuralNet n(topology);
    n.feedForward({1,2,3,4,5});
    std::cout << "HELLO" << std::endl;
    std::cout << n << std::endl;
}

void run_neural_tests() {
    test_feed_forward();
}

int main() {
    run_neural_tests();
    return 0;
}
