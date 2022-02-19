package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;

import java.util.HashMap;
import java.util.Map;

/**
 * A class to represent an OVA classifier implemented with a ClassifierFactory
 *
 * Prepared for CS158 Assignment 05. Authored by David D'Attile
 */
public class OVAClassifier implements Classifier{
    private final ClassifierFactory factory;
    private final Map<Double, Classifier> labelClassifierMap;

    /**
     * Initialize the OVA classifier. The classifier relies on the passed
     * ClassifierFactory for hyperparameters. At initialization/without training
     * this classifier yields classifications of 'NaN' for passed examples.
     *
     * @param factory
     */
    public OVAClassifier (ClassifierFactory factory) {
        this.factory = factory;
        this.labelClassifierMap = new HashMap<>();
    }

    /**
     * Train this classifier based on the data set
     *
     * @param data
     */
    @Override
    public void train(DataSet data) {

        // clear classifier map if retraining
        this.labelClassifierMap.clear();

        // loop through amount of labels
        for (double currLabel : data.getLabels()) {

            // create classifier
            Classifier currClassifier = this.factory.getClassifier();

            // create binary copy of data
            DataSet binaryLabelSet = createBinaryDatasetCopy(data, currLabel);

            // train classifier on binary data
            currClassifier.train(binaryLabelSet);

            labelClassifierMap.put(currLabel, currClassifier);
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

        // init max confidence and associated label
        double maxPositiveConfidence = Double.MIN_VALUE;
        double minNegativeConfidence = Double.MAX_VALUE;
        double posConfidentLabel = Double.NaN;
        double negConfidentLabel = Double.NaN;

        // loop through binary classifiers and classify
        for (double currLabel : this.labelClassifierMap.keySet()) {

            // get classifier, classify, and get classification confidence
            Classifier currClassifier = this.labelClassifierMap.get(currLabel);
            double currClassification = currClassifier.classify(example);
            double currConfidence = currClassifier.confidence(example);

            // if we make positive classification and are confident, update
            if (currClassification > 0) {
                if (currConfidence > maxPositiveConfidence) {
                    maxPositiveConfidence = currConfidence;
                    posConfidentLabel = currLabel;
                }
            } else { // otherwise, update if least confident negative classification
                if (currConfidence < minNegativeConfidence) {
                    minNegativeConfidence = currConfidence;
                    negConfidentLabel = currLabel;
                }
            }
        }

        // return positive confident label if we saw a positive example, otherwise negative confident label
        return maxPositiveConfidence != Double.MIN_VALUE ? posConfidentLabel : negConfidentLabel;
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
     * Helper function that creates a binary copy of the provided dataset where
     * the label specified is changed to '1.0' and all others are changed to '-1.0'
     *
     * @param dataSet
     * @param positiveLabel
     * @return the new binary data set representation
     */
    private static DataSet createBinaryDatasetCopy(DataSet dataSet, double positiveLabel) {

        // make new dataset based on provided dataset's feature map
        DataSet binaryDataSet = new DataSet(dataSet.getFeatureMap());

        // add modified examples based on original data set's examples
        for (Example e : dataSet.getData()) {
            Example binaryExample = new Example(e);

            // transition to binary example based on provided label
            if (binaryExample.getLabel() == positiveLabel) {
                binaryExample.setLabel(1.0);
            } else {
                binaryExample.setLabel(-1.0);
            }

            // add binary example to binary set
            binaryDataSet.addData(binaryExample);
        }

        // return new binary dataset
        return binaryDataSet;
    }
}
