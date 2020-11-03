
#include "nn.h"

NN::NN(const std::function<double(double)> activation, 
       const size_t inputs, 
       const size_t outputs, 
       const std::uniform_real_distribution<double>& dis, 
       const std::mt19937& gen,
       size_t& innovOn,
       const struct MutationConfig& config,
       const double activationLevel) :
         activation(activation),
         dis(dis),
         gen(gen),
         activationLevel(activationLevel),
         innovOn(innovOn) {
  this->inputLayer.size = inputs;
  this->outputLayer.size = outputs;
  for (int o = 0; o < outputs; ++o) {
    this->outputLayer.nodes.push_back(o + inputs);
    for (int i = 0; i < inputs; ++i) {
      struct Gene gene(i + o * inputs, i, o + inputs);
      this->genotype.insert(gene);
      this->inputLayer.nodes.push_back(i);
      this->incoming[o + inputs].insert(i);
      this->weights[{i, o + inputs}] = this->dis(this->gen);
    }
  }
  this->configMutations(config);
}

void NN::configMutations(const struct MutationConfig& config) {
  double sum = 0;
  for (double next : {config.toggleGene, 
                      config.addConn, 
                      config.addNode, 
                      config.randomizeWeight}) {
    sum += next;
    this->mutationCdf.push_back(sum);
  }
  this->mutationDis = std::uniform_real_distribution<double>(0, sum);
}

std::vector<double> NN::propagate(const std::vector<double>& input) {
  if (input.size() != this->inputLayer.size) {
    throw std::runtime_error("NN input layer size does not match input vector size");
  }

  std::map<const size_t, double> memo;
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

double NN::propagateRecurse(std::map<const node, double>& memo, const node& nodeOn) {
  std::unordered_set<node> invNeighbors = this->incoming[nodeOn];
  double sum = 0;
  for (const node& nodeN : invNeighbors) {
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

void NN::insertGene(const struct Gene& gene) {
  this->insertGene(gene, this->dis(this->gen));
}

void NN::insertGene(const struct Gene& gene, double weight) {
  this->genotype.insert(gene);
  this->incoming[gene.out].insert(gene.in);
  this->weights[{gene.in, gene.out}] = this->dis(this->gen);
}

void NN::disableGene(const size_t innov) {
  struct Gene gene = this->getGene(innov);
  this->incoming[gene.out].erase(gene.in);
  gene.enabled = false;
}

void NN::toggleGene(const size_t innov) {
  struct Gene gene = this->getGene(innov);
  if (gene.enabled) {
    this->incoming[gene.out].erase(gene.in);
    gene.enabled = false;
  } else {
    this->incoming[gene.out].insert(gene.in);
    gene.enabled = true;
  }
}

void NN::addConn(const size_t innov, const node from, const node to) {
  struct Gene gene(innov, from, to);
  this->insertGene(gene);
}

void NN::addNode(const size_t innov1, 
                 const size_t innov2, 
                 const size_t oldInnov, 
                 const node newNode) {
  struct Gene oldGene = this->getGene(oldInnov);
  node from = oldGene.in;
  node to = oldGene.out;
  struct Gene gene1(innov1, from, newNode);
  struct Gene gene2(innov2, newNode, to);
  this->insertGene(gene1, this->activationLevel);
  this->insertGene(gene2, this->getWeight(oldInnov));
  this->disableGene(oldInnov);
}

void NN::randomizeWeight(const size_t innov) {
  struct Gene gene = this->getGene(innov);
  this->weights[{gene.in, gene.out}] = this->dis(this->gen);
}

const struct Gene& NN::getGene(const size_t innov) {
  auto i = this->genotype.find(innov);
  if (i != this->genotype.end()) {
    return *i;
  }
  throw std::runtime_error("Gene with innovation number not found");
}

double NN::getWeight(const size_t innov) {
  const struct Gene gene = this->getGene(innov);
  return this->weights[{gene.in, gene.out}];
}
