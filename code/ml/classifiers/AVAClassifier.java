package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;

/**
 * A class to represent an OVA classifier implemented with a ClassifierFactory
 *
 * Prepared for CS158 Assignment 05. Authored by David D'Attile
 */
public class AVAClassifier implements Classifier{
    private final ClassifierFactory factory;

    /**
     * Initialize the AVA classifier. The classifier relies on the passed
     * ClassifierFactory for hyperparameters. At initialization/without training
     * this classifier yields classifications of 'NaN' for passed examples.
     *
     * @param factory
     */
    public AVAClassifier (ClassifierFactory factory) {
        this.factory = factory;
    }

    /**
     * Train this classifier based on the data set
     *
     * @param data
     */
    @Override
    public void train(DataSet data) {

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

    /**
     * TODO: implement when necessary
     * @param example
     * @return
     */
    @Override
    public double confidence(Example example) {
        return 0;
    }

    /**
     * A helper function that takes a data set, a label to set as 1.0 (pos), and a
     * label to set as -1.0 (neg) and returns a trimmed copy of the input data set that
     * contains only these two classes
     *
     * @param dataSet
     * @param posLabel
     * @param negLabel
     * @return the trimmed copy of the data set with classes 1.0 and -1.0
     */
    private static DataSet dataSetTrimmedCopy(DataSet dataSet, double posLabel, double negLabel) {
        return null;
    }
}
