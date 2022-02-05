package ml.classifiers;

import ml.DataSet;
import ml.Example;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class AveragePerceptronClassifier extends PerceptronClassifier implements Classifier {

    private Double avgBias;
    private ArrayList<Double> avgWeights;
    private int totalCounter;

    /**
     * Initialize the average perceptron classifier. At initialization, the perceptron
     * defaults to a prediction of '0.0', has a max iterations value of 10,
     * and has no default dataSet.
     */
    public AveragePerceptronClassifier() {
        this.avgBias = 0.0;
        this.avgWeights = new ArrayList<>();
        this.totalCounter = 1;
    }

    /**
     * Train this classifier based on the data set. If training data set is empty,
     * examples will be classified with label '0.0'.
     *
     * @param dataSet training data
     */
    @Override
    @SuppressWarnings("unchecked")
    // TODO: this has not been changed from reg perceptron
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

                // isolate prediction and feature label
                double cLabel = classify(e);
                double fLabel = e.getLabel();

                // update if prediction based on current model is wrong
                if (cLabel * fLabel <= 0) {

                    // update all weights with wi = wi + (fi * fLabel)
                    for (int j = 0; j < this.weights.size(); j++) {
                        double updatedWeight = this.weights.get(j) + (e.getFeature(j) * fLabel);
                        this.weights.set(j, updatedWeight);
                    }

                    // update bias with actual label
                    this.bias += fLabel;
                }
            }
        }
    }

    /**
     * Classify the example based on average weights and bias.
     * Should only be called *after* train has been called.
     *
     * @param example an example from a test data set
     * @return the class label predicted by the classifier for this example (1 or -1)
     */
    @Override
    // TODO: double check that we only compared against avg weights!
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

        // w1f1 + w2f2 + ... + wnfn based on average weights
        for (int i = 0; i < weightCount; i++) {
            classification += this.avgWeights.get(i) * example.getFeature(i);
        }

        // add avg bias
        classification += this.avgBias;

        // TODO: check that < is okay
        // return 1 or -1 based on classification sign
        if (classification < 0) {
            return -1.0;
        } else {
            return 1.0;
        }
    }

    /**
     * Helper method for resetting a perceptron's regular and avg. weights/bias to 0.0
     *
     * @param weightCount amount of weights initialized to 0.0 needed
     */
    protected void reset(Integer weightCount) {
        // reset biases to zero
        this.bias = 0.0;
        this.avgBias = 0.0;

        // reset regular and avg weights to zero
        this.weights.clear();
        this.avgWeights.clear();

        for (int i = 0; i < weightCount; i++) {
            this.weights.add(0.0);
            this.avgWeights.add(0.0);
        }
    }
}
