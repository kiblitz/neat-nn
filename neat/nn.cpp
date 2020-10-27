
#include "nn.h"

NN::NN(std::function<double(double)> activation, 
       int inputs, 
       int outputs, 
       std::uniform_real_distribution<double>& dis, 
       std::mt19937& gen) {
  this->activation = activation;
  this->inputLayer.size = inputs;
  this->outputLayer.size = outputs;
  for (int o = 0; o < outputs; ++o) {
    this->outputLayer.nodes.push_back(o + inputs);
    for (int i = 0; i < inputs; ++i) {
      struct Gene gene;
      gene.innov = i + o * inputs;
      gene.in = i;
      gene.out = o + inputs;
      this->genotype.push_back(gene);
      this->inputLayer.nodes.push_back(i);
      this->incoming[o + inputs].push_back(i);
      this->weights[{i, o + inputs}] = dis(gen);
    }
  }
}

double NN::propagateRecurse(std::map<int, double>& memo, const int& node) {
  std::vector<int> invNeighbors = this->incoming[node];
  double sum = 0;
  for (int i = 0; i < invNeighbors.size(); ++i) {
    int nodeN = invNeighbors[i];
    if (memo.count(nodeN) == 0) {
      memo[nodeN] = propagateRecurse(memo, nodeN);
    }
    sum += memo[nodeN] * weights[{nodeN, node}]; 
  }
  return this->activation(sum);
}

std::vector<double> NN::propagate(const std::vector<double>& input) {
  if (input.size() != this->inputLayer.size) {
    throw std::runtime_error("NN input layer size does not match input vector size");
  }

  std::map<int, double> memo;
  for (int i = 0; i < this->inputLayer.size; ++i) {
    int node = this->inputLayer.nodes[i];
    memo[node] = input[i];
  }
  
  std::vector<double> output;
  for (int i = 0; i < this->outputLayer.size; ++i) {
    output.push_back(propagateRecurse(memo, this->outputLayer.nodes[i]));
  }
  return output;
}
