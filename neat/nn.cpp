
#include "nn.h"

NN::NN(std::function<double(double)> activation, 
       size_t inputs, 
       size_t outputs, 
       std::uniform_real_distribution<double>& dis, 
       std::mt19937& gen,
       double activationLevel) {
  this->dis = dis;
  this->gen = gen;
  this->activationLevel = activationLevel;
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
      this->incoming[o + inputs].insert(i);
      this->weights[{i, o + inputs}] = this->dis(this->gen);
    }
  }
}

std::vector<double> NN::propagate(const std::vector<double>& input) {
  if (input.size() != this->inputLayer.size) {
    throw std::runtime_error("NN input layer size does not match input vector size");
  }

  std::map<size_t, double> memo;
  for (int i = 0; i < this->inputLayer.size; ++i) {
    node nodeOn = this->inputLayer.nodes[i];
    memo[nodeOn] = input[i];
  }
 
  std::vector<double> output;
  for (int i = 0; i < this->outputLayer.size; ++i) {
    output.push_back(propagateRecurse(memo, this->outputLayer.nodes[i]));
  }
  return output;
}

double NN::propagateRecurse(std::map<node, double>& memo, const node& nodeOn) {
  std::unordered_set<node> invNeighbors = this->incoming[nodeOn];
  double sum = 0;
  for (node nodeN : invNeighbors) {
    if (memo.count(nodeN) == 0) {
      memo[nodeN] = propagateRecurse(memo, nodeN);
    }
    sum += memo[nodeN] * weights[{nodeN, nodeOn}]; 
  }
  if (invNeighbors.size() == 0) {
    sum = this->activationLevel;
  }
  return this->activation(sum);
}

void NN::insertGene(struct Gene& gene) {
  this->insertGene(gene, this->dis(this->gen));
}

void NN::insertGene(struct Gene& gene, double weight) {
  for (auto i = this->genotype.rbegin(); i != this->genotype.rend(); ++i) {
    size_t innov = i->innov;
    if (innov == gene.innov) {
      this->incoming[i->out].insert(i->in);
      i->enabled = true;
      return;
    }
    if (innov < gene.innov) {
      this->incoming[gene.out].insert(gene.in);
      this->weights[{gene.in, gene.out}] = weight;
      this->genotype.insert(i.base(), gene);
      return;
    }
  }
  this->incoming[gene.out].insert(gene.in);
  this->weights[{gene.in, gene.out}] = this->dis(this->gen);
  this->genotype.insert(genotype.begin(), gene);
}

void NN::disableGene(size_t innov) {
  struct Gene gene = this->getGene(innov);
  this->incoming[gene.out].erase(gene.in);
  gene.enabled = false;
}

void NN::toggleGene(size_t innov) {
  struct Gene gene = this->getGene(innov);
  if (gene.enabled) {
    this->incoming[gene.out].erase(gene.in);
    gene.enabled = false;
  } else {
    this->incoming[gene.out].insert(gene.in);
    gene.enabled = true;
  }
}

void NN::addConn(size_t innov, node from, node to) {
  struct Gene gene;
  gene.innov = innov;
  gene.in = from;
  gene.out = to;
  this->insertGene(gene);
}

void NN::addNode(size_t innov1, size_t innov2, size_t oldInnov, 
                 node from, node between, node to) {
  struct Gene gene1;
  gene1.innov = innov1;
  gene1.in = from;
  gene1.out = between;
  struct Gene gene2;
  gene2.innov = innov2;
  gene2.in = between;
  gene2.out = to;
  this->insertGene(gene1, this->activationLevel);
  this->insertGene(gene2, this->getWeight(oldInnov));
  this->disableGene(oldInnov);
}

struct Gene& NN::getGene(size_t innov) {
  for (auto i = this->genotype.begin(); i != this->genotype.end(); ++i) {
    if (i->innov == innov) {
      return *i;
    }
  } 
  throw std::runtime_error("Gene with innovation number not found");
}

double NN::getWeight(size_t innov) {
  struct Gene gene = this->getGene(innov);
  return this->weights[{gene.in, gene.out}];
}
