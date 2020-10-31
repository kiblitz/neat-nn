
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

typedef size_t node;

struct Gene {
  size_t innov;
  node in;
  node out;
  bool enabled = true;
};

struct Layer {
  std::vector<node> nodes;
  size_t size;
};

class NN {
  public:
    std::vector<Gene> genotype; 
    std::map<std::pair<node, node>, double> weights;
    Layer inputLayer;
    Layer outputLayer;
    std::map<node, std::unordered_set<node>> incoming;
  
    std::function<double(double)> activation; 
    std::uniform_real_distribution<double> dis;
    std::mt19937 gen;

    /** 
     * Recursive propagation helper
     *
     * @param memo Memoization map with propagated values of each node
     * @param nodeOn Node currently on in recursion
     */
    double propagateRecurse(std::map<node, double>& memo, const node& nodeOn);

    /** 
     * Inserts gene to genotype
     *
     * Adds random weight and updates incoming map. If gene in genotype, 
     * just enable it 
     *
     * @param gene Gene to insert
     */
    void insertGene(struct Gene& gene);
    
    /** 
     * Disables a connection
     *
     * @param innov Innovation of gene with connection to disable
     */
    void disableGene(size_t innov);

    /** Adds a new connection between given nodes
     *
     * @param innov Innovation number of gene corresponding to new connection
     * @param from Connection start node
     * @param to Connection end node
     */
    void addConn(size_t innov, node from, node to);

    /** Adds a new node between given nodes
     *
     * @param innov1 Innovation number of gene corresponding to first new connection
     * @param innov2 Innovation number of gene corresponding to second new connection
     * @param oldInnov Innovation number of gene corresponding to old connection
     * @param from Connection start node
     * @param between New node
     * @param to Connection end node
     */   
    void addNode(size_t innov1, size_t innov2, size_t oldInnov, 
                 node from, node between, node to);
  public:
    NN(std::function<double(double)> activation, 
       size_t inputs, 
       size_t outputs, 
       std::uniform_real_distribution<double>& dis, 
       std::mt19937& gen);

    std::vector<double> propagate(const std::vector<double>& input);

    void mutate();
    NN& breed(const NN& mate);
};
