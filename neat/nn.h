
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

typedef size_t node;

// Neural network gene
struct Gene {
  size_t innov;
  node in;
  node out;
  bool enabled = true;
};

// Neural network layer
struct Layer {
  std::vector<node> nodes;
  size_t size;
};

// Probabilities configuration
struct MutationConfig {
  MutationConfig(): toggleGene(0.25),
                    addConn(0.25),
                    addNode(0.25),
                    randomizeWeight(0.25) {}
  double toggleGene;
  double addConn;
  double addNode;
  double randomizeWeight;
};

// Mutation constants
enum MUTATION {
  TOGGLEGENE,
  ADDCONN,
  ADDNODE,
  RANDOMIZEWEIGHT
};

class NN {
  public:
    // Vector of genes corresponding to neural network genotype
    std::vector<Gene> genotype; 

    // Map with connection keys to weight values
    std::map<std::pair<node, node>, double> weights;

    // Input layer
    Layer inputLayer;

    // Output layer
    Layer outputLayer;

    // Map with node keys to incoming nodes
    std::map<node, std::unordered_set<node>> incoming;
  
    // Activation function
    std::function<double(double)> activation; 

    // Weight distribution
    std::uniform_real_distribution<double> dis;

    // Random number generator
    std::mt19937 gen;

    // Random mutation cdf
    std::vector<double> mutationCdf;
    
    // Mutation distribution
    std::uniform_real_distribution<double> mutationDis;

    // Value for biases and node insertions
    double activationLevel = 1;

    /** 
     * Recursive propagation helper
     *
     * @param memo Memoization map with propagated values of each node
     * @param nodeOn Node currently on in recursion
     */
    double propagateRecurse(std::map<node, double>& memo, const node& nodeOn);

    /**
     * Configure mutation probabilities from config
     *
     * @param config Mutation configuration to use
     */
    void configMutations(const struct MutationConfig& config);
    
    /** 
     * Inserts gene to genotype
     *
     * Adds random weight and updates incoming map. If gene in genotype, 
     * just enable it 
     *
     * @param gene Gene to insert
     * @param weight Weight of connection indicated by gene
     */
    void insertGene(struct Gene& gene);
    void insertGene(struct Gene& gene, double weight);
    
    /** 
     * Disables a connection
     *
     * @param innov Innovation of gene with connection to disable
     */
    void disableGene(size_t innov);

    /**
     * Toggles a connection
     *
     * @param innov Innovation of gene with connection to toggle
     */
    void toggleGene(size_t innov);

    /** 
     * Adds a new connection between given nodes
     *
     * @param innov Innovation number of gene corresponding to new connection
     * @param from Connection start node
     * @param to Connection end node
     */
    void addConn(size_t innov, node from, node to);

    /** 
     * Adds a new node between given nodes
     *
     * @param innov1 Innovation number of gene for first new connection
     * @param innov2 Innovation number of gene for second new connection
     * @param oldInnov Innovation number of gene for old connection
     * @param newNode New node to add
     */   
    void addNode(size_t innov1, size_t innov2, size_t oldInnov, node newNode);

    /**
     * Randomizes a weight given an innovation number
     *
     * @param innov Innovation number of gene for connection to randomize weight
     */
    void randomizeWeight(size_t innov);

    /**
     * Get the gene corresponding to the given innovation number
     *
     * @param innov Innovation number of gene
     * @return The gene corresponding to the given innovation number
     */
    struct Gene& getGene(size_t innov);

    /**
     * Get the weight of a connection from its gene innovation number
     *
     * @param innov Innovation number of gene for connection
     * @return The weight of the connection associated with the given gene
     */
    double getWeight(size_t innov);

  public:
    /**
     * Neural network constructor
     *
     * @param activation Activation function
     * @param inputs Number of input nodes
     * @param outputs Number of output nodes
     * @param dis Distribution for possible weights
     * @param gen Random number generator for weights
     * @param config Mutation probabilities configuration
     * @param activationLevel Set the activation level value (default = 1)
     */
    NN(std::function<double(double)> activation, 
       size_t inputs, 
       size_t outputs, 
       std::uniform_real_distribution<double>& dis, 
       std::mt19937& gen,
       const struct MutationConfig& config = MutationConfig(),
       double activationValue = 1);

    /**
     * Propagate neural network
     *
     * @param input Vector of inputs to propagate through neural network
     * @return Vector of outputs as a result of propagation
     */
    std::vector<double> propagate(const std::vector<double>& input);

    void mutate();
    NN& breed(const NN& mate);
};
