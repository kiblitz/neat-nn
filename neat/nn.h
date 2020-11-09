
#include <functional>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

typedef size_t node;
typedef size_t innovNum;
typedef std::pair<node, node> connection;

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
    // Set of innovation numbers corresponding to neural network genotype
    std::set<innovNum> genotype; 

    // Map with innovation numbers to gene enabled
    std::map<innovNum, bool> enabledGenes;

    // Map with connection keys to weight values
    std::map<connection, double> weights;

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

    // Map of existing connections to innovation numbers in environment gene pool
    std::map<connection, innovNum>& connPool;

    // Map of innovation numbers to existing connections in environment gene pool
    std::map<innovNum, connection>& genePool;

    // Next innovation number
    innovNum& innovOn;

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
     * Inserts gene innovation number to genotype
     *
     * Adds random weight and updates incoming map. If gene in genotype, 
     * just enable it 
     *
     * @param innov Innovation number of gene to insert
     * @param conn Connection of gene to insert 
     * @param weight Weight of connection indicated by gene
     * @param enabled Gene inserted enabled characteristic
     */
    void insertGene(const innovNum innov, 
                    const connection conn,
                    const bool enabled = true);
    void insertGene(const innovNum innov, 
                    const connection conn,
                    const double weight, 
                    const bool enabled = true);
 
    /** 
     * Disables a connection
     *
     * @param innov Innovation of gene with connection to disable
     */
    void disableGene(const innovNum innov);

    /**
     * Toggles a connection
     *
     * @param innov Innovation of gene with connection to toggle
     */
    void toggleGene(const innovNum innov);

    /** 
     * Adds a new connection between nodes in a given connection
     *
     * @param innov Innovation number of gene corresponding to new connection
     * @param conn Connection of gene to add node between
     */
    void addConn(const innovNum innov, const connection conn);

    /** 
     * Adds a new node between given nodes
     *
     * @param innov1 Innovation number of gene for first new connection
     * @param innov2 Innovation number of gene for second new connection
     * @param oldInnov Innovation number of gene for old connection
     * @param newNode New node to add
     */   
    void addNode(const innovNum innov1, 
                 const innovNum innov2, 
                 const innovNum oldInnov, 
                 const node newNode);

    /**
     * Randomizes a weight given an innovation number
     *
     * @param innov Innovation number of gene for connection to randomize weight
     */
    void randomizeWeight(const innovNum innov);

    /**
     * Get the weight of a connection from its gene innovation number
     *
     * @param innov Innovation number of gene for connection
     * @return The weight of the connection associated with the given gene
     */
    double getWeight(const innovNum innov);

  public:
    /**
     * Neural network constructor
     *
     * @param activation Activation function
     * @param inputs Number of input nodes
     * @param outputs Number of output nodes
     * @param dis Distribution for possible weights
     * @param gen Random number generator for weights
     * @param connPool Map of all connections to innovation numbers
     * @param genePool Map of all innovation numbers to connections
     * @param innovOn Next innovation number to append to gene pool
     * @param config Mutation probabilities configuration
     * @param activationLevel Set the activation level value (default = 1)
     */
    NN(const std::function<double(double)> activation, 
       const size_t inputs, 
       const size_t outputs, 
       const std::uniform_real_distribution<double>& dis, 
       const std::mt19937& gen,
       std::map<connection, innovNum>& connPool,
       std::map<innovNum, connection>& genePool,
       innovNum& innovOn,
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
