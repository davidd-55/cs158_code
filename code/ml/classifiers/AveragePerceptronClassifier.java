package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;
import java.util.*;

/**
 * A class to represent an average perceptron classifier.
 * This class extends the regular perceptron class and only modifies the 'train' method.
 *
 * Prepared for CS158 Assignment 03. Authored by David D'Attile
 */
public class AveragePerceptronClassifier extends PerceptronClassifier implements Classifier {
    /**
     * Train this classifier based on the data set. If training data set is empty,
     * examples will be classified with label '0.0'.
     *
     * @param dataSet training data
     */
    @Override
    @SuppressWarnings("unchecked")
    public void train(DataSet dataSet) {
        // reset weights and bias to zero
        this.reset(dataSet.getAllFeatureIndices().size());

        // get weight counts
        int weightCount = this.weights.size();

        // isolate and clone data for training
        ArrayList<Example> clonedData = (ArrayList<Example>)dataSet.getData().clone();

        // initialize updated and totalCount
        int updated = 0;
        int totalCounter = 0;

        // initialize average weights/bias
        double avgBias = 0.0;
        ArrayList<Double> avgWeights = new ArrayList<>(this.weights);

        // train for maxIterations iterations
        for (int i = 0; i < maxIterations; i++) {

            // shuffle training data
            Collections.shuffle(clonedData, new Random(System.nanoTime()));

            // loop through shuffled data for training
            for (Example e : clonedData) {

                // update if prediction based on current model is wrong
                if (classify(e) * e.getLabel() <= 0) {

                    // update our final, weighted, avg weights
                    for (int j = 0; j < weightCount; j++) {
                        double updatedAvgWeight = avgWeights.get(j) + (updated * this.weights.get(j));
                        avgWeights.set(j, updatedAvgWeight);
                    }

                    // update avg bias
                    avgBias += updated * this.bias;

                    // update all weights with wi = wi + (fi * label)
                    for (int k = 0; k < weightCount; k++) {
                        double updatedWeight = this.weights.get(k) + (e.getFeature(k) * e.getLabel());
                        this.weights.set(k, updatedWeight);
                    }

                    // update bias with actual label
                    this.bias += e.getLabel();

                    // reset updated
                    updated = 0;
                }

                // update updated and total counter regardless of model update
                updated += 1;
                totalCounter += 1;
            }
        }

        // reset weights/bias instance vars with averaged versions
        for (int i = 0; i < weightCount; i++) {
            double avgWeight = avgWeights.get(i) / totalCounter;
            this.weights.set(i, avgWeight);
        }

        this.bias = avgBias / totalCounter;
    }

    /**
     * TODO: implement if necessary
     * @param example
     * @return
     */
    @Override
    public double confidence(Example example) {
        return 0;
    }
}
