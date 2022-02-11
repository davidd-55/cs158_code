package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;
import ml.data.ExampleWithDistance;

import java.util.ArrayList;

/**
 * A class to represent a K Nearest Neighbor classifier
 *
 * Prepared for CS158 Assignment 04. Authored by David D'Attile
 */
public class KNNClassifier implements Classifier {
    private int k;
    private ArrayList<Example> comparisonExamples;

    /**
     * Initialize the KNN classifier. At initialization, the classifier
     * defaults to a prediction of '0.0', has a K-value of 3, and comparisonExample.
     */
    public KNNClassifier() {
        this.k = 3;
        this.comparisonExamples = new ArrayList<>();
    }

    /**
     * Saves the data to compare a provided example to. Explicitly, the classifier
     * clones the provided data examples.
     *
     * @param data
     */
    @Override
    @SuppressWarnings("unchecked")
    public void train(DataSet data) {
        this.comparisonExamples = (ArrayList<Example>) data.getData().clone();
    }

    /**
     * Classify the example. Should only be called *after* train has been called.
     *
     * @param example
     * @return the class label predicted by the classifier for this example
     */
    @Override
    public double classify(Example example) {
        return 0;
    }

    /**
     * Sets the K-value for the KNN classifier
     *
     * @param k set K-value for the KNN classifier. Throws an error for
     *          non-positive inputs.
     */
    public void setK(int k) {
        // throw error for non-positive input K's
        if (k <= 0) {
            String msg = String.format("expected a positive K-value; received %d", k);
            throw new IllegalArgumentException(msg);
        }
        this.k = k;
    }

    private double calculateDistance(Example e1, Example e2) {
        int e1FeatureCount = e1.getFeatureSet().size();
        int e2FeatureCount = e2.getFeatureSet().size();

        if (e1)

        for (int i = 0; i < )
    }
}
