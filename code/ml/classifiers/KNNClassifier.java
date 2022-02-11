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

    /**
     * Calculates the euclidian distance between the features of two examples. Throws an
     * error if the two examples have a different feature count.
     *
     * @param e1
     * @param e2
     * @return a double representing the euclidian distance between the two given examples
     */
    private double calculateDistance(Example e1, Example e2) {
        // init feature counts
        int e1FeatureCount = e1.getFeatureSet().size();
        int e2FeatureCount = e2.getFeatureSet().size();

        // basic error check for incompatible features
        if (e1 != e2) {
            String msg = String.format(
                    "cannot calculate distance between two examples with feature counts %d (e1), %d (e2)",
                    e1FeatureCount,
                    e2FeatureCount);
            throw new IllegalArgumentException(msg);
        }

        // calculate distance except final sqrt
        double d = 0.0;
        for (int i = 0; i < e1FeatureCount; i++) {
            d += Math.pow(e1.getFeature(i) - e2.getFeature(i), 2);
        }

        // return final distance
        return Math.sqrt(d);
    }
}
