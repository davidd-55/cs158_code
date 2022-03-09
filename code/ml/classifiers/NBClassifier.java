package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;

import java.util.HashMap;

/**
 * A class to represent a Naive Bayes classifier with all feature and
 * positive feature only classification.
 *
 * Prepared for CS158 Assignment 07b. Authored by David D'Attile
 */
public class NBClassifier implements Classifier {
    private double lambda; // used as smoothing param
    private boolean useOnlyPositiveFeatures;
    private long exampleCount;
    private HashMapCounter<Double> labelOccurrences;
    private HashMapCounter<Integer> allFeatureOccurrences;
    private HashMap<Double,HashMapCounter<Integer>> labelFeatureOccurrences;

    /**
     * Initialize the Naive Bayes classifier. At initialization/without training
     * this classifier yields classifications of 'NaN' for passed examples. This implementation
     * initializes lambda as BLANK and by default uses all features.
     */
    public NBClassifier() {
        this.lambda = 0.01;
        this.useOnlyPositiveFeatures = false;
        this.exampleCount = 0;
        this.labelOccurrences = new HashMapCounter<>();
        this.allFeatureOccurrences = new HashMapCounter<>();
        this.labelFeatureOccurrences = new HashMap<>();
    }

    /**
     * Train this classifier based on the data set
     *
     * @param data
     */
    @Override
    public void train(DataSet data) {
        // reset feature count data
        this.exampleCount = data.getData().size();
        this.labelOccurrences.clear();
        this.allFeatureOccurrences.clear();
        this.labelFeatureOccurrences.clear();

        // loop through examples
        for (Example e : data.getData()) {
            // increment label occurrences
            double label = e.getLabel();
            this.labelOccurrences.increment(label);

            // loop through available features for each example
            for (int featureNum : e.getFeatureSet()) {
                // increment feature count if it occurs in this example
                this.allFeatureOccurrences.increment(featureNum); // increment total count
                this.labelFeatureOccurrences.get(label).increment(featureNum); // increment per label count
            }
        }
    }

    /**
     * Classify the example.  Should only be called *after* train has been called.
     *
     *
     * @param example
     * @return the class label predicted by the classifier for this example
     */
    @Override
    public double classify(Example example) {
        return 0;
    }

    /**
     * Calculates the log probability of the most likely label (i.e. the
     * label predicted for that example).
     *
     * @param example
     * @return A double representing the probability of the most likely label
     */
    @Override
    public double confidence(Example example) {
        // get prediction
        double prediction = classify(example);

        // calculate log prob
        return getLogProb(example, prediction);
    }

    /**
     * Calculates the log (base 10 ) probability of the example with the label under
     * the current trained model, i.e. log(p(f1, f2,..., fm, label)).
     *
     * @param ex
     * @param label
     * @return a double representation of the log probability.
     */
    public double getLogProb(Example ex, double label) {
        // use helper fxns based on useOnlyPositiveFeatures
        return this.useOnlyPositiveFeatures ? getPosFeaturesLogProb(ex, label) : getAllFeaturesLogProb(ex, label);
    }

    /**
     * Calculates probability of a feature with the label under
     * the current trained model, i.e. p(feature | label).
     *
     * @param featureIndex
     * @param label
     * @returna double representation of the feature probability.
     */
    public double getFeatureProb(int featureIndex, double label) {
        // get label and all occurrences of a feature
        double labelOccurrences = this.labelFeatureOccurrences.get(label).get(featureIndex);
        double allOccurrences = this.allFeatureOccurrences.get(featureIndex);

        // divide!
        return labelOccurrences / allOccurrences;
    }

    /**
     * A helper fxn for calculating the positive features only log probability,
     * i.e. p(f1, f2, ..., fm, y) for all features f in a given example.
     *
     * @param ex
     * @param label
     * @return a double representation of the all features log prob. calculation
     */
    private double getAllFeaturesLogProb(Example ex, double label) {
        // init sum as log(prob. of label)
        double probabilitySum = Math.log10(getLabelProbability(label));

        // loop through all feature indices
        for (int featureIndex : this.allFeatureOccurrences.keySet()) {
            // get p(feature | label)
            double featureProbability = getFeatureProb(featureIndex, label);

            // if example contains feature, log(featureProbability),
            // otherwise, log(1 - featureProbability).
            probabilitySum += ex.getFeatureSet().contains(featureIndex)
                    ? Math.log10(featureProbability)
                    : Math.log10(1 - featureProbability);
        }

        return probabilitySum;
    }

    /**
     * A helper fxn for calculating the positive features only log probability,
     * i.e. p(f1, f2, ..., fm, y) for all positive features f in a given example.
     *
     * @param ex
     * @param label
     * @return a double representation of the positive features log prob. calculation
     */
    private double getPosFeaturesLogProb(Example ex, double label) {
        // init sum as log(prob. of label)
        double probabilitySum = Math.log10(getLabelProbability(label));

        // sum log(p(feature | label))
        for (int featureIndex : ex.getFeatureSet()) {
            probabilitySum += Math.log10(getFeatureProb(featureIndex, label));
        }

        return probabilitySum;
    }

    /**
     * A helper fxn for calculating the probability of a label, i.e.
     * label occurrences / total example count.
     *
     * @param label
     * @return double representation of the label probability
     */
    private double getLabelProbability(double label) {
        return this.labelOccurrences.get(label) / (double) this.exampleCount;
    }

    /**
     * A helper fxn for setting the value of lambda.
     *
     * @param lambda
     */
    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    /**
     * A helper fxn for setting the value of useOnlyPositiveFeatures.
     *
     * @param useOnlyPositiveFeatures
     */
    public void setUseOnlyPositiveFeatures(boolean useOnlyPositiveFeatures) {
        this.useOnlyPositiveFeatures = useOnlyPositiveFeatures;
    }
}
