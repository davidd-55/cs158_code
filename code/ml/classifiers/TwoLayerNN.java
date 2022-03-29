package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;

import java.util.ArrayList;
import java.util.Collections;

public class TwoLayerNN implements Classifier {

    // constants for the different activation functions
    public static final int TANH_ACTIVATION = 0;
    public static final int SIGMOID_ACTIVATION = 1;

    // set min/max random weight vals
    private static final double MIN_RANDOM_WEIGHT = -0.1;
    private static final double MAX_RANDOM_WEIGHT = 0.1;

    // network-specific instance vars
    private final int numHiddenNodes;
    private int activationFxn;
    private double eta; // aka learning rate
    private double iterations;

    // set up weight matrices
    private double[][] firstLayerWeights;
    private double[] hiddenLayerWeights; // one-dimensional since we assume 1 output

    public TwoLayerNN(int numHiddenNodes) {
        this.numHiddenNodes = numHiddenNodes;
        this.eta = 0.1;
        this.iterations = 200;
        this.activationFxn = TANH_ACTIVATION;
    }

    /**
     * Train this classifier based on the data set
     *
     * @param data
     */
    @Override
    @SuppressWarnings("unchecked")
    public void train(DataSet data) {
        // TODO: implement bias

        // init network weights
        // TODO: maybe bias here?
        int featureCount = data.getFeatureMap().size();
        this.initializeWeights(featureCount);

        // get data
        ArrayList<Example> training = (ArrayList<Example>)data.getData().clone();

        // loop through iterations
        for (int i = 0; i < this.iterations; i++) {
            // shuffle data
            Collections.shuffle(training);
        }
    }

    /**
     * Classify the example.  Should only be called *after* train has been called.
     *
     * @param example
     * @return the class label predicted by the classifier for this example
     */
    @Override
    public double classify(Example example) {
        return 0;
    }

    @Override
    public double confidence(Example example) {
        return 0;
    }

    /**
     * A helper fxn for calculating the specified activation fxn's result for a given input.
     *
     * @param input
     * @return a double representing the result of the specified activation fxn
     */
    private double calculateActivation(double input) {
        // tanh activation
        if (this.activationFxn == TANH_ACTIVATION) {
            return Math.tanh(input);
        } else { // sigmoid activation
            return 1 / (1 + Math.exp(-1 * input)); // a?
        }
    }

    /**
     * Set the eta (learning rate) the NN should use
     *
     * @param eta
     */
    public void setEta(double eta){
        this.eta = eta;
    }

    /**
     * Set the iterations the NN should use during training
     *
     * @param iterations
     */
    public void setIterations(int iterations){
        this.iterations = iterations;
    }

    /**
     * A helper function for setting the network's activation fxn to tanh.
     */
    public void setTanhActivation() {
        this.activationFxn = TANH_ACTIVATION;
    }

    /**
     * A helper function for setting the network's activation fxn to sigmoid.
     */
    public void setSigmoidActivation() {
        this.activationFxn = SIGMOID_ACTIVATION;
    }

    /**
     * An internal helper function for checking if the NN's activation function is tanh.
     */
    private boolean isTanhActivation() {
        return this.activationFxn == TANH_ACTIVATION;
    }

    /**
     * An internal helper function for checking if the NN's activation function is sigmoid.
     */
    private boolean isSigmoidActivation() {
        return this.activationFxn == SIGMOID_ACTIVATION;
    }

    /**
     * Initialize the network's weights based on the number of features in an example.
     * Each weight will be randomized between MIN_RANDOM_WEIGHT and MAX_RANDOM_WEIGHT.
     *
     * @param featureCount
     */
    private void initializeWeights(int featureCount) {
        // m (feature count) * d (num. hidden nodes) matrix
        this.firstLayerWeights = new double[featureCount][this.numHiddenNodes];

        // d (num. hidden nodes) * o (num. outputs = 1) matrix
        this.hiddenLayerWeights = new double[this.numHiddenNodes];

        // init random weight values in first layer weights
        for (double[] currRow : this.firstLayerWeights) {
            for (int i = 0; i < currRow.length; i++) {
                currRow[i] = getRandomWeightValue();
            }
        }

        // init random weight values in hidden layer weights
        for (int i =0; i < this.hiddenLayerWeights.length; i++) {
            this.hiddenLayerWeights[i] = getRandomWeightValue();
        }
    }

    /**
     * A helper fxn for calculating a random weight value for the initialization
     * of the two layer NN class.
     *
     * @return a double represent
     */
    private static double getRandomWeightValue() {
        // return random value between min and max
        return (Math.random() * ((MAX_RANDOM_WEIGHT - MIN_RANDOM_WEIGHT) + 1)) + MIN_RANDOM_WEIGHT;
    }
}
