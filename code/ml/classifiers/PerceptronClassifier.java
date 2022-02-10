package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;
import java.util.*;

/**
 * A class to represent a perceptron classifier
 *
 * Prepared for CS158 Assignment 03. Authored by David D'Attile
 */
public class PerceptronClassifier implements Classifier {
    protected Integer maxIterations;
    protected Double bias;
    protected ArrayList<Double> weights;

    /**
     * Initialize the perceptron classifier. At initialization, the perceptron
     * defaults to a prediction of '0.0', has a max iterations value of 10,
     * and has no default dataSet.
     */
    public PerceptronClassifier() {
        this.maxIterations = 10;
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

        // get weight counts
        int weightCount = this.weights.size();

        // train for maxIterations iterations
        for (int i = 0; i < maxIterations; i++) {

            // shuffle training data
            Collections.shuffle(clonedData, new Random(System.nanoTime()));

            // loop through shuffled data for training
            for (Example e : clonedData) {

                // update if prediction based on current model is wrong
                if (classify(e) * e.getLabel() <= 0) {

                    // update all weights with wi = wi + (fi * label)
                    for (int j = 0; j < weightCount; j++) {
                        double updatedWeight = this.weights.get(j) + (e.getFeature(j) * e.getLabel());
                        this.weights.set(j, updatedWeight);
                    }

                    // update bias with actual label
                    this.bias += e.getLabel();
                }
            }
        }
    }

    /**
     * Classify the example.  Should only be called *after* train has been called.
     *
     * @param example an example from a test data set
     * @return the class label predicted by the classifier for this example (1 or -1)
     */
    @Override
    public double classify(Example example) {
        // get number of weights
        int weightCount = this.weights.size();

        // check for weights count == feature count
        if (weightCount != example.getFeatureSet().size()) {
            String msg = String.format(
                    "expected example feature count to match model. example feature count: %d model feature count: %d",
                    example.getFeatureSet().size(),
                    weightCount);

            throw new IllegalArgumentException(msg);
        }

        // init classification
        double classification = 0.0;

        // w1f1 + w2f2 + ... + wnfn
        for (int i = 0; i < weightCount; i++) {
            classification += this.weights.get(i) * example.getFeature(i);
        }

        // add bias
        classification += this.bias;

        // TODO: check that < is okay
        // return 1 or -1 based on classification sign
        if (classification < 0) {
            return -1.0;
        } else {
            return 1.0;
        }
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
            String msg = String.format("expected a positive 'iterations' value, received: %d", iterations);
            throw new IllegalArgumentException(msg);
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