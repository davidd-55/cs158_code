package ml.data;

/**
 * A class that normalizes data set examples such that each example has a length of 1.
 *
 * Prepared for CS158 Assignment 04. Authored by David D'Attile
 */
public class ExampleNormalizer implements DataPreprocessor{
    /**
     * Preprocess the training data by normalizing each example's length to 1.
     *
     * @param train
     */
    @Override
    public void preprocessTrain(DataSet train) {
        for (Example e : train.getData()) {
            normalizeExample(e);
        }
    }

    /**
     * Preprocess the testing data by normalizing each example's length to 1.
     *
     * @param test
     */
    @Override
    public void preprocessTest(DataSet test) {
        for (Example e : test.getData()) {
            normalizeExample(e);
        }
    }

    /**
     * A helper function for normalizing a feature such that it's length is 1
     * after this function is called.
     *
     * @param e
     */
    private static void normalizeExample(Example e) {
        // get example length
        double l = calculateFeatureLength(e);

        // calculate new feature value for each feature based on
        // the example's length
        for (int i = 0; i < e.getFeatureSet().size(); i++) {
            double normalizedFeature = e.getFeature(i) / l;
            e.setFeature(i, normalizedFeature);
        }
    }

    /**
     * A helper function for calculating an example's length
     *
     * @param e
     * @return a double representing the example's length
     */
    private static double calculateFeatureLength(Example e) {
        // init length
        double length = 0.0;

        // add value of each feature squared to length
        for (int i = 0; i < e.getFeatureSet().size(); i++) {
            length += Math.pow(e.getFeature(i), 2);
        }

        // return sqrt of length
        return Math.sqrt(length);
    }
}
