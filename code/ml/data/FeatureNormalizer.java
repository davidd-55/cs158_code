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
     * Preprocess the training data
     *
     * @param train
     */
    @Override
    public void preprocessTrain(DataSet train) {

    }

    /**
     * Preprocess the testing data
     *
     * @param test
     */
    @Override
    public void preprocessTest(DataSet test) {

    }
}
