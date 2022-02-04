package ml.classifiers;

import ml.DataSet;
import ml.Example;

public class AveragePerceptronClassifier extends PerceptronClassifier implements Classifier {
    /**
     * Train this classifier based on the data set. If training data set is empty,
     * examples will be classified with label '0.0'.
     *
     * @param dataSet training data
     */
    @Override
    public void train(DataSet dataSet) {
        // reset weights and biases
        this.reset(dataSet.getAllFeatureIndices().size());

    }

    /**
     * Classify the example.  Should only be called *after* train has been called.
     *
     * @param example an example from a test data set
     * @return the class label predicted by the classifier for this example
     */
    @Override
    public double classify(Example example) {
        return 0;
    }
}
