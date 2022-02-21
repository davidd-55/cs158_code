package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;

import java.util.*;

/**
 * A class to represent an OVA classifier implemented with a ClassifierFactory
 *
 * Prepared for CS158 Assignment 05. Authored by David D'Attile
 */
public class AVAClassifier implements Classifier{
    private final ClassifierFactory factory;
    private final List<Double> labels;
    // in the format (posLabelIndex, negLabelIndex) --> classifier
    private final Map<List<Integer>, Classifier> labelIndicesClassifierMap;

    /**
     * Initialize the AVA classifier. The classifier relies on the passed
     * ClassifierFactory for hyperparameters. At initialization/without training
     * this classifier yields classifications of 'NaN' for passed examples.
     *
     * @param factory
     */
    public AVAClassifier (ClassifierFactory factory) {
        this.labels = new ArrayList<>();
        this.factory = factory;
        this.labelIndicesClassifierMap = new HashMap<>();
    }

    /**
     * Train this classifier based on the data set
     *
     * @param data
     */
    @Override
    public void train(DataSet data) {

        // clear old labels and map
        this.labels.clear();
        this.labelIndicesClassifierMap.clear();

        // get label count & list representation
        this.labels.addAll(data.getLabels());
        int labelCount = this.labels.size();

        // loop through examples
        for (int i = 0; i < labelCount; i++) {
            for (int j = i + 1; j < labelCount; j++) {

                // isolate pos and neg values
                double posLabel = this.labels.get(i);
                double negLabel = this.labels.get(j);
                final List<Integer> currLabelIndices = List.of(i, j);

                // init classifier
                Classifier currClassifier = this.factory.getClassifier();

                // build trimmed binary data set
                DataSet trimmedBinaryDataSet = dataSetBinaryTrimmedCopy(data, posLabel, negLabel);

                // train classifier based on trimmed data set
                currClassifier.train(trimmedBinaryDataSet);

                // put labels and classifier into map
                this.labelIndicesClassifierMap.put(currLabelIndices, currClassifier);
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

        // init label count
        int labelCount = this.labels.size();

        // intialize score array with all zeros
        double[] scoreArray = new double[labelCount];
        Arrays.fill(scoreArray, 0.0);

        // loop through label indices --> classifier map
        for (List<Integer> labelIndices : this.labelIndicesClassifierMap.keySet()) {

            // get indicies and classifier
            int i = labelIndices.get(0);
            int j = labelIndices.get(1);
            Classifier currClassifier = this.labelIndicesClassifierMap.get(labelIndices);

            // get prediction and confidence and mult. them together
            double prediction = currClassifier.classify(example);
            double confidence = currClassifier.confidence(example);
            double score = prediction * confidence;

            // increment scores
            scoreArray[i] += score;
            scoreArray[j] -= score;
        }

        // init max index/score
        int maxIndex = Integer.MIN_VALUE;
        double maxScore = Double.MIN_VALUE;

        // update max index/score
        for (int scoreIndex = 0; scoreIndex < labelCount; scoreIndex++) {
            double currScore = scoreArray[scoreIndex];
            if (currScore > maxScore) {
                maxIndex = scoreIndex;
                maxScore = currScore;
            }
        }

        return this.labels.get(maxIndex);
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
    private static DataSet dataSetBinaryTrimmedCopy(DataSet dataSet, double posLabel, double negLabel) {

        // make new dataset based on provided dataset's feature map
        DataSet binaryDataSet = new DataSet(dataSet.getFeatureMap());

        // add modified examples based on original data set's examples
        for (Example e : dataSet.getData()) {

            // get label
            double label = e.getLabel();

            // add to new data set only if one of our desired labels
            if (label == posLabel || label == negLabel) {

                // transition to binary example based on provided label; add to data set
                double newLabel = label == posLabel ? 1.0 : -1.0;
                binaryDataSet.addData(new Example(e){{setLabel(newLabel);}});
            }
        }

        // return new binary dataset
        return binaryDataSet;
    }
}
