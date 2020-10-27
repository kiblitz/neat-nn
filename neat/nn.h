
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

struct Gene {
  int innov;
  int in;
  int out;
  bool enabled = true;
};

struct Layer {
  std::vector<int> nodes;
  size_t size;
};

class NN {
  public:
    std::vector<Gene> genotype; 
    std::map<std::pair<int, int>, double> weights;
    Layer inputLayer;
    Layer outputLayer;
    std::map<int, std::vector<int>> incoming;
  
    std::function<double(double)> activation; 

    double propagateRecurse(std::map<int, double>& memo, const int& node);

    NN addNode();
    NN addConn();
  //public:
    NN(std::function<double(double)> activation, 
       int inputs, 
       int outputs, 
       std::uniform_real_distribution<double>& dis, 
       std::mt19937& gen);
    std::vector<double> propagate(const std::vector<double>& input);
    NN mutate();
    NN breed(const NN& mate);
};
