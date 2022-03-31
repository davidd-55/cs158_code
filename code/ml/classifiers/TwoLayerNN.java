package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class TwoLayerNN implements Classifier {

    // constants for the different activation functions
    public static final int TANH_ACTIVATION = 0;
    public static final int SIGMOID_ACTIVATION = 1;

    // randomizer for initial weight values
    private static final Random RANDOM = new Random();

    // network-specific instance vars
    private final int numHiddenNodes;
    private int activationFxn;
    private double eta; // aka learning rate
    private double iterations;
    private boolean includeBias;

    // set up weight matrices
    private double[][] hiddenWeights;
    private double[] outputWeights; // one-dimensional since we assume 1 output

    // make hidden layer values after forward propagation accessible
    private double[] hiddenLayerValues;

    public TwoLayerNN(int numHiddenNodes) {
        this.numHiddenNodes = numHiddenNodes;
        this.eta = 0.1;
        this.iterations = 200;
        this.activationFxn = TANH_ACTIVATION;
        this.includeBias = true;
    }

    /**
     * Train this classifier based on the data set
     *
     * @param data
     */
    @Override
    @SuppressWarnings("unchecked")
    public void train(DataSet data) {
        // determine whether we need to generate a dataset with bias feature
        DataSet dataToTrain = this.includeBias ? data.getCopyWithBias() : data;

        // init network weights and hidden values
        int featureCount = dataToTrain.getFeatureMap().size();
        initializeWeights(featureCount);
        initializeHiddenValues();

        // get data
        ArrayList<Example> training = (ArrayList<Example>)dataToTrain.getData().clone();

        // loop through iterations
        for (int i = 0; i < this.iterations; i++) {
            // shuffle data
            Collections.shuffle(training);

            // loop through examples
            for (Example e : training) {
                // compute example e through the network and update newly
                // computed hidden layer values
                double prediction = forwardCompute(e);

                // backpropagation
                backpropagation(e, prediction);
            }
        }
    }

    /**
     * Initialize the network's weights based on the number of features in an example.
     * Each weight will be randomized between MIN_RANDOM_WEIGHT and MAX_RANDOM_WEIGHT.
     *
     * @param featureCount
     */
    private void initializeWeights(int featureCount) {
        // init bias value
        int biasValue = this.includeBias ? this.numHiddenNodes + 1 : this.numHiddenNodes;

        // d (num. hidden nodes) * m (feature count) matrix
        this.hiddenWeights = new double[this.numHiddenNodes][featureCount];

        // d (num. hidden nodes) * o (num. outputs = 1) matrix; +1 for bias weight
        this.outputWeights = new double[biasValue];

        // init random weight values in hidden layer weights
        for (double[] currRow : this.hiddenWeights) {
            for (int i = 0; i < currRow.length; i++) {
                currRow[i] = getRandomWeightValue();
            }
        }

        // init random weight values in output layer weights
        for (int i =0; i < this.outputWeights.length; i++) {
            this.outputWeights[i] = getRandomWeightValue();
        }
    }

    /**
     * Initialize the network's hidden node values based on the desired number
     * of hidden nodes. Include a hard-coded bias of 1 if specified.
     */
    private void initializeHiddenValues() {
        // init bias value based on number of hidden nodes, +1 if true
        int biasValue = this.includeBias ? this.numHiddenNodes + 1 : this.numHiddenNodes;

        // init hidden layer value array to all 0's
        this.hiddenLayerValues = new double[biasValue];

        // if using bias, set final array value to 1
        if (this.includeBias) {
            this.hiddenLayerValues[this.hiddenLayerValues.length - 1] = 1.0;
        }
    }

    /**
     * Perform the forward computation of an example through the NN.
     *
     * @param e
     * @return a double representing the forward computation result
     */
    private double forwardCompute(Example e) {
        // get feature array from example
        double[] featureValues = getFeatureArray(e);

        // get hidden layer node count; -1 for bias since we don't want
        // to change hard coded 1 bias node
        int hiddenLayerNodeCount = this.includeBias ?
                this.hiddenLayerValues.length - 1:
                this.hiddenLayerValues.length;

        // first layer calculation loop
        for (int i = 0; i < hiddenLayerNodeCount; i++) {
            // get all feature weights for a first layer node d
            double[] hiddenNodeWeights = this.hiddenWeights[i];

            // save activation of dot product calc (feature array . hidden layer node weight array)
            this.hiddenLayerValues[i] = calculateActivation(dotProduct(featureValues, hiddenNodeWeights));
        }

        // hidden layer dot product calc (hidden layer vals 'v' dot output layer weight array 'h')
        return calculateActivation(dotProduct(this.hiddenLayerValues,this.outputWeights));
    }

    /**
     * Perform the backwards propagation of error through the network's weights from the
     * output of the network for a given example.
     */
    private void backpropagation(Example e, double prediction) {
        // get label
        double label = e.getLabel();

        // loop through/update output weights first; get f'(v . h)
        double vDotHDerivative = calculateActivationDerivative(
                dotProduct(this.hiddenLayerValues,this.outputWeights));
        for (int i =0; i < this.outputWeights.length; i++) {
            // get current weight and hidden node value
            double currWeight = this.outputWeights[i];
            double currHiddenVal = this.hiddenLayerValues[i];

            // TODO: is hk the node value with activation already computed?
            // weight update: vk = vk + (eta * hk * (label - prediction) * f'(v . h))
            this.outputWeights[i] = currWeight + (this.eta * currHiddenVal * (label - prediction) * vDotHDerivative);
        }

        // then loop through/update hidden weights
        for (double[] currRow : this.hiddenWeights) {
            for (int i = 0; i < currRow.length; i++) {
                // currRow[i] = getRandomWeightValue();
            }
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
        // TODO: sigmoid 0 instead of -1.0
        return forwardCompute(example) > 0 ? 1.0 : -1.0;
    }

    /**
     * Gives the NN's confidence in its prediction for a given example.
     * Here, confidence is the absolute value of the result of forward-computing an example
     * through the network.
     *
     * @param example
     * @return a double representing the NN's confidence
     */
    @Override
    public double confidence(Example example) {
        return Math.abs(forwardCompute(example));
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
            return 1 / (1 + Math.exp(-1 * input));
        }
    }

    /**
     * A helper fxn for calculating the specified activation fxn's derivative result for a given input.
     *
     * @param input
     * @return a double representing the derivative of the result of the specified activation fxn
     */
    private double calculateActivationDerivative(double input) {
        // tanh activation derivative; t'(x) = 1 - (t(x)^2)
        if (this.activationFxn == TANH_ACTIVATION) {
            return 1 - Math.pow(Math.tanh(input), 2);
        } else { // sigmoid activation derivative; s'(x) = s(x) * (1 - s(x))
            return (1 / (1 + Math.exp(-1 * input)) * (1 - (1 / (1 + Math.exp(-1 * input)))));
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
     * A helper function for setting whether the network should include a bias in training.
     *
     * @param includeBias
     */
    public void setIncludeBias(boolean includeBias) { this.includeBias = includeBias; }

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
     * A helper fxn for calculating a dot product between two arrays.
     *
     * @param a1
     * @param a2
     * @return a double representing the dot product
     */
    private static double dotProduct(double[] a1, double[] a2) {
        // if lengths un equal, throw exception
        if (a1.length != a2.length) {
            String msg = String.format("Cannot calculate dot product for arrays of different lengths. a1:%d a2:%d", a1.length, a2.length);
            throw new IllegalArgumentException(msg);
        }

        // loop through vals and calc dot product
        double dp = 0.0;
        for (int i = 0; i < a1.length; i++) {
            dp += a1[i] * a2[i];
        }

        return dp;
    }

    /**
     * Generate an array of doubles representing an example's feature values
     *
     * @param e
     * @return a double array containing an examples features
     */
    private static double[] getFeatureArray(Example e) {
        // get feature count
        int featureCount = e.getFeatureSet().size();

        // init feature array
        double[] featureArray = new double[featureCount];

        // set array vals
        for (int i = 0; i < featureCount; i++) {
            featureArray[i] = e.getFeature(i);
        }

        return featureArray;
    }

    /**
     * A helper fxn for calculating a random weight value for the initialization
     * of the two layer NN class.
     *
     * @return a double represent
     */
    private static double getRandomWeightValue() {
        // return random value between -0.1 and 0.1
        return (RANDOM.nextBoolean() ? 1 : -1) * (RANDOM.nextDouble() * 0.1);
    }

    /**
     * A helper fxn for nicely printing a double array.
     *
     * @param array
     */
    private static void printArray(double[] array) {
        // init string builder
        StringBuilder s = new StringBuilder("[ ");

        // append values to string
        for (int i = 0; i < array.length; i++) {
            double val = array[i];

            // edge case for last val
            if (i < array.length - 1) {
                s.append(String.format("%f, ", val));
            } else {
                s.append(String.format("%f", val));
            }
        }

        System.out.println(s.append(" ]"));
    }
}
