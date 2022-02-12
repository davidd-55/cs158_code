package ml.data;

import java.util.ArrayList;

/**
 * A class that normalizes data set features such that each feature is divided
 * by the training set mean and standard deviation for that feature.
 *
 * Prepared for CS158 Assignment 04. Authored by David D'Attile
 */
public class FeatureNormalizer implements DataPreprocessor{
    private ArrayList<Double> trainFeatureMeans;
    private ArrayList<Double> trainFeatureStandardDeviations;

    /**
     * Initialize the Feature Normalizer data preprocessor. At initialization, the preprocessor
     * defaults initializes empty training feature means and stamdard deviations.
     */
    public FeatureNormalizer() {
        this.trainFeatureMeans = new ArrayList<>();
        this.trainFeatureStandardDeviations = new ArrayList<>();
    }

    /**
     * Preprocess the training data by subtracting a feature's mean from itself
     * then dividing by the feature's standard deviation.
     *
     * @param train
     */
    @Override
    public void preprocessTrain(DataSet train) {

        // get feature count
        int featureCount = train.getFeatureMap().size();

        // calculate feature means and standard deviations
        for (int i = 0; i < featureCount; i++) {
            double featureMean = calculateFeatureMean(train, i);
            this.trainFeatureMeans.add(featureMean);

            double featureVariance = calculateFeatureVariance(train, i, featureMean);
            this.trainFeatureStandardDeviations.add(featureVariance);
        }

        // for each example, center then adjust w/variance scaling
        for (Example e : train.getData()) {
            for (int i = 0; i < featureCount; i++) {
                double newFeatureValue = (e.getFeature(i) - this.trainFeatureMeans.get(i)) / this.trainFeatureStandardDeviations.get(i);
                e.setFeature(i, newFeatureValue);
            }
        }
    }

    /**
     * Preprocess the testing data by subtracting a feature's mean from itself
     * then dividing by the feature's standard deviation.
     *
     * @param test
     */
    @Override
    public void preprocessTest(DataSet test) {

        // get feature count
        int featureCount = test.getFeatureMap().size();

        // for each example, center then adjust w/variance scaling
        for (Example e : test.getData()) {
            for (int i = 0; i < featureCount; i++) {
                double newFeatureValue = (e.getFeature(i) - this.trainFeatureMeans.get(i)) / this.trainFeatureStandardDeviations.get(i);
                e.setFeature(i, newFeatureValue);
            }
        }
    }

    /**
     * A helper function for determining the average value of features at the specified index.
     *
     * @param train
     * @param featureIndex
     * @return the average value of features at the specified index
     */
    private static double calculateFeatureMean(DataSet train, int featureIndex) {

        // init sum
        double sum = 0.0;

        // loop through features and add to sum
        for (Example e : train.getData()) {
            sum += e.getFeature(featureIndex);
        }

        // return feature mean
        return sum / train.getData().size();
    }

    /**
     * A helper function for determining the standard deviation of features
     * at the specified index.
     *
     * @param train
     * @param featureIndex
     * @return the standard deviation of features at the specified index
     */
    private static double calculateFeatureVariance(
            DataSet train,
            int featureIndex,
            double featureMean) {

        // init sum
        double sum = 0.0;

        // loop through features and add (feature - mean)^2 to sum
        for (Example e : train.getData()) {
            sum += Math.pow((e.getFeature(featureIndex) - featureMean), 2);
        }

        // return feature std dev
        return Math.sqrt(sum / train.getData().size());
    }
}
