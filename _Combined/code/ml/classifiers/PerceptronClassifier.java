package ml.classifiers;

import ml.DataSet;
import ml.Example;
import java.util.*;

/**
 * A class to represent a perceptron classifier
 *
 * Prepared for CS158 Assignment 03. Authored by David D'Attile
 */
public class PerceptronClassifier implements Classifier {
    private Integer maxIterations;
    private Double bias;
    private ArrayList<Double> weights;

    /**
     * Initialize the decision tree classifier. At initialization, the
     * decision tree is a decision stump defaulting to a prediction of '0.0',
     * has no depth limit, and has no default dataSet.
     */
    public PerceptronClassifier() {
        this.maxIterations = 0;
        this.bias = 0.0;
        this.weights = new ArrayList<>();
    }

    /**
     * Train this classifier based on the data set. If training data set is empty,
     * examples will be classified with label '0.0'.
     *
     * @param dataSet training data
     */
    @Override
    @SuppressWarnings("unchecked")
    public void train(DataSet dataSet) {
        // reset weights and bias to zero
        this.reset(dataSet.getAllFeatureIndices().size());

        // isolate and clone data for training
        ArrayList<Example> clonedData = (ArrayList<Example>)dataSet.getData().clone();

        // train for maxIterations iterations
        for (int i = 0; i < maxIterations; i++) {

            // shuffle training data
            Collections.shuffle(clonedData, new Random(System.nanoTime()));

            // loop through shuffled data for training
            for (Example e : clonedData) {

            }
        }
    }

    /**
     * Classify the example.  Should only be called *after* train has been called.
     *
     * @param example an example from a test data set
     * @return the class label predicted by the classifier for this example
     */
    @Override
    public double classify(Example example) {
        // TODO: check for weights count == feature count

        // 0 = w1f1 + w2f2 + ... + wnfn + b

        //double sum
        return 0.0;
    }

    /**
     * Helper method for resetting a perceptron's weights/bias to 0.0
     *
     * @param weightCount amount of weights initialized to 0.0 needed
     */
    protected void reset(Integer weightCount) {
        // reset bias to zero
        this.bias = 0.0;

        // reset weights to zero
        this.weights.clear();
        for (int i = 0; i < weightCount; i++) {
            this.weights.add(0.0);
        }
    }

    /**
     * Sets the maximum number of training iterations for the perceptron classifier
     *
     * @param iterations iterations to set for the perceptron classifier;
     *                   throws InvalidArgumentException for arguments less than zero
     */
    public void setIterations(int iterations) {
        // throw error for supplied iterations < 0
        if (iterations < 0) {
            String message = String.format("expected a positive 'iterations' value, received: %d", iterations);
            throw new IllegalArgumentException(message);
        }

        this.maxIterations = iterations;
    }

    /**
     * Generates a string representation of the perceptron weights and bias
     * in the format '0:weight_0 1:weight_1 2:weight2 ... n:weight_n b-value'
     *
     * @return the string representation of the perceptron
     */
    public String toString() {
        // if weights haven't been set or are empty, return empty string
        if (this.weights.isEmpty()) {
            return "";
        }

        // init StringBuilder
        StringBuilder s = new StringBuilder();

        // add weights to string
        for (int i = 0; i < this.weights.size(); i++) {
            s.append(String.format("%d:%f ", i, this.weights.get(i)));
        }

        // add bias
        s.append(String.format("%f", this.bias));

        return s.toString();
    }
}