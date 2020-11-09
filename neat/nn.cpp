
#include "nn.h"

NN::NN(const std::function<double(double)> activation, 
       const size_t inputs, 
       const size_t outputs, 
       const std::uniform_real_distribution<double>& dis, 
       const std::mt19937& gen,
       std::map<std::pair<node, node>, innovNum>& connPool,
       std::map<innovNum, std::pair<node, node>>& genePool,
       size_t& innovOn,
       const struct MutationConfig& config,
       const double activationLevel) :
         activation(activation),
         dis(dis),
         gen(gen),
         activationLevel(activationLevel),
         connPool(connPool),
         genePool(genePool),
         innovOn(innovOn) {
  this->inputLayer.size = inputs;
  this->outputLayer.size = outputs;
  for (size_t o = 0; o < outputs; ++o) {
    this->outputLayer.nodes.push_back(o + inputs);
    for (size_t i = 0; i < inputs; ++i) {
      this->inputLayer.nodes.push_back(i);
      const connection conn = {i, o + inputs};
      this->insertGene(i + o * inputs, conn, this->dis(this->gen));
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
  for (size_t i = 0; i < this->inputLayer.size; ++i) {
    const node nodeOn = this->inputLayer.nodes[i];
    memo[nodeOn] = input[i];
  }
 
  std::vector<double> output;
  for (size_t i = 0; i < this->outputLayer.size; ++i) {
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

void NN::insertGene(const innovNum innov, 
                    const connection conn, 
                    bool enabled) {
  this->insertGene(innov, conn, this->dis(this->gen), enabled);
}

void NN::insertGene(const innovNum innov,
                    const connection conn,
                    double weight, 
                    bool enabled) {
  if (innov > this->innovOn) {
    throw std::runtime_error("Inserting gene with innovation number out of bounds");
  }
  if (innov == this->innovOn) {
    this->innovOn++;
    this->connPool[conn] = innov;
    this->genePool[innov] = conn;
  } else if (this->genePool[innov] != conn) {
    throw std::runtime_error("Inserting gene with wrong corresponding " 
                             "innovation number and connection");
  }
  if (enabled) {
    this->incoming[conn.second].insert(conn.first);
  }
  this->enabledGenes[innov] = enabled;
  this->genotype.insert(innov);
  this->weights[conn] = weight;
}

void NN::disableGene(const size_t innov) {
  const connection conn = this->genePool[innov];
  this->incoming[conn.second].erase(conn.first);
  this->enabledGenes[innov] = false;
}

void NN::toggleGene(const size_t innov) {
  const connection conn = this->genePool[innov];
  if (enabledGenes[innov]) {
    this->incoming[conn.second].erase(conn.first);
  } else {
    this->incoming[conn.second].insert(conn.first);
  }
  this->enabledGenes[innov] = !this->enabledGenes[innov];
}

void NN::addConn(const size_t innov, const connection conn) {
  this->insertGene(innov, conn);
}

void NN::addNode(const size_t innov1, 
                 const size_t innov2, 
                 const size_t oldInnov, 
                 const node newNode) {
  const connection oldConn = this->genePool[oldInnov];
  const connection conn1 = {oldConn.first, newNode};
  const connection conn2 = {newNode, oldConn.second};
  this->insertGene(innov1, conn1, this->activationLevel);
  this->insertGene(innov2, conn2, this->getWeight(oldInnov));
  this->disableGene(oldInnov);
}

void NN::randomizeWeight(const size_t innov) {
  const connection conn = this->genePool[innov];
  this->weights[conn] = this->dis(this->gen);
}

double NN::getWeight(const size_t innov) {
  const connection conn = this->genePool[innov];
  return this->weights[conn];
}
