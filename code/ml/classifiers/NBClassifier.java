package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

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
    private Set<Integer> featureIndices;
    private HashMapCounter<Double> labelOccurrences;
    // private HashMapCounter<Integer> allFeatureOccurrences;
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
        this.featureIndices = new HashSet<>();
        this.labelOccurrences = new HashMapCounter<>();
        // this.allFeatureOccurrences = new HashMapCounter<>();
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
        this.featureIndices = data.getAllFeatureIndices();
        this.labelOccurrences.clear();
        // this.allFeatureOccurrences.clear();
        this.labelFeatureOccurrences.clear();

        // loop through examples
        for (Example e : data.getData()) {
            // increment label occurrences
            double label = e.getLabel();
            this.labelOccurrences.increment(label);

            // loop through available features for each example
            for (int featureNum : e.getFeatureSet()) {

                // this.allFeatureOccurrences.increment(featureNum); // increment total count

                // if label not in label feature occurrences map, put it!
                if (!labelFeatureOccurrences.containsKey(label)) {
                    labelFeatureOccurrences.put(label, new HashMapCounter<>());
                }

                // increment feature count if it occurs in this example
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
        // init max probability and label
        double maxProbability = Double.NEGATIVE_INFINITY;
        double maxProbabilityLabel = Double.NaN;

        for (double label : this.labelOccurrences.keySet()) {
            // get log probability of example and label
            double logProbabilityOfEx = getLogProb(example, label);

            // if log prob. of example > previous, reset max probability and label
            if (logProbabilityOfEx > maxProbability) {
                maxProbability = logProbabilityOfEx;
                maxProbabilityLabel = label;
            }
        }

        return maxProbabilityLabel;
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
     * A helper fxn for calculating the positive features only log probability,
     * i.e. p(f1, f2, ..., fm, y) for all positive features f in a given example.
     *
     * @param ex
     * @param label
     * @return a double representation of the positive features log prob. calculation
     */
    private double getPosFeaturesLogProb(Example ex, double label) {
        // init example probability sum as log(prob. of label)
        double exampleProbability = Math.log10(getLabelProbability(label));

        // sum log(p(feature | label))
        for (int featureIndex : ex.getFeatureSet()) {
            // summation of log(p(feature | label))
            exampleProbability += Math.log10(getFeatureProb(featureIndex, label));
        }

        return exampleProbability;
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
        // init example probability sum as log(prob. of label)
        double exampleProbability = Math.log10(getLabelProbability(label));

        // loop through all feature indices
        for (int featureIndex : this.featureIndices) {
            // get p(feature | label)
            double featureProbability = getFeatureProb(featureIndex, label);

            // if example contains feature, add log(featureProbability),
            // otherwise, add log(1 - featureProbability).
            exampleProbability += ex.getFeatureSet().contains(featureIndex)
                    ? Math.log10(featureProbability)
                    : Math.log10(1 - featureProbability);
        }

        return exampleProbability;
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
        double labelFeatureOccurrences = this.labelFeatureOccurrences.get(label).get(featureIndex);
        double labelOccurrences = this.labelOccurrences.get(label);

        // calculate smoothed prob! -> count(feature, label) + lambda / count(label) + possible_feature_vals * lambda
        return (labelFeatureOccurrences + this.lambda) / (labelOccurrences + (this.featureIndices.size() * this.lambda));
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
