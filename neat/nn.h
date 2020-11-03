
#include <functional>
#include <map>
#include <random>
#include <set>
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

  Gene(const size_t& innov) : innov(innov) {}
  Gene(const size_t& innov, const node& in, const node& out) : 
         innov(innov), 
         in(in), 
         out(out) {}

  bool operator<(const struct Gene& gene) const {
    return innov < gene.innov;
  }
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
    // Set of genes corresponding to neural network genotype
    std::set<struct Gene> genotype; 

    // Map with innovation numbers to gene enabled
    std::map<size_t, bool> enabledGenes;

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

    // Gene pool with all genes
    std::set<struct Gene>& genePool;

    // Next innovation number
    size_t& innovOn;

    /** 
     * Recursive propagation helper
     *
     * @param memo Memoization map with propagated values of each node
     * @param nodeOn Node currently on in recursion
     */
    double propagateRecurse(std::map<const node, double>& memo, const node& nodeOn);

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
    void insertGene(const struct Gene& gene);
    void insertGene(const struct Gene& gene, double weight);
 
    /** 
     * Disables a connection
     *
     * @param innov Innovation of gene with connection to disable
     */
    void disableGene(const size_t innov);

    /**
     * Toggles a connection
     *
     * @param innov Innovation of gene with connection to toggle
     */
    void toggleGene(const size_t innov);

    /** 
     * Adds a new connection between given nodes
     *
     * @param innov Innovation number of gene corresponding to new connection
     * @param from Connection start node
     * @param to Connection end node
     */
    void addConn(const size_t innov, const node from, const node to);

    /** 
     * Adds a new node between given nodes
     *
     * @param innov1 Innovation number of gene for first new connection
     * @param innov2 Innovation number of gene for second new connection
     * @param oldInnov Innovation number of gene for old connection
     * @param newNode New node to add
     */   
    void addNode(const size_t innov1, 
                 const size_t innov2, 
                 const size_t oldInnov, 
                 const node newNode);

    /**
     * Randomizes a weight given an innovation number
     *
     * @param innov Innovation number of gene for connection to randomize weight
     */
    void randomizeWeight(const size_t innov);

    /**
     * Get the gene corresponding to the given innovation number
     *
     * @param innov Innovation number of gene
     * @return The gene corresponding to the given innovation number
     */
    const struct Gene& getGene(const size_t innov);

    /**
     * Checks if gene corresponding to the given innovation number exists
     *
     * @param innov Innovation number of gene
     * @return Whether or not the gene ixists
     */
    bool hasGene(const size_t innov);

    /**
     * Get the weight of a connection from its gene innovation number
     *
     * @param innov Innovation number of gene for connection
     * @return The weight of the connection associated with the given gene
     */
    double getWeight(const size_t innov);

  public:
    /**
     * Neural network constructor
     *
     * @param activation Activation function
     * @param inputs Number of input nodes
     * @param outputs Number of output nodes
     * @param dis Distribution for possible weights
     * @param gen Random number generator for weights
     * @param genePool Set of all genes
     * @param innovOn Next innovation number to append to gene pool
     * @param config Mutation probabilities configuration
     * @param activationLevel Set the activation level value (default = 1)
     */
    NN(const std::function<double(double)> activation, 
       const size_t inputs, 
       const size_t outputs, 
       const std::uniform_real_distribution<double>& dis, 
       const std::mt19937& gen,
       std::set<struct Gene>& genePool,
       size_t& innovOn,
       const struct MutationConfig& config = MutationConfig(),
       const double activationLevel = 1);

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
