package ml;

import ml.classifiers.*;
import ml.data.*;

import java.util.ArrayList;
import java.util.List;

/**
 * A class to run Naive Bayes classifier experiments
 *
 * Prepared for CS158 Assignment 07b. Authored by David D'Attile
 */
public class Experimenter {

    /**
     * Main method for running the experiments specified.
     * @param args
     */
    public static void main(String[] args) {

        // init datasets
        DataSet titanicData = new DataSet("/Users/daviddattile/Dev/cs158_code/data/titanic-train.csv", DataSet.CSVFILE);
        DataSetSplit titanicSplit= titanicData.split(0.8);
        //CrossValidationSet wineXV = wineData.getCrossValidationSet(10);

        // init NN classifiers
        TwoLayerNN twoLayerNNNoBias = new TwoLayerNN(3);
        twoLayerNNNoBias.setIncludeBias(false);
        TwoLayerNN twoLayerNNWithBias = new TwoLayerNN(3);
        twoLayerNNWithBias.setIncludeBias(true);

        twoLayerNNNoBias.train(titanicSplit.getTrain());
        twoLayerNNWithBias.train(titanicSplit.getTrain());

        System.out.println("lambda,allFeatsAccuracy,posFeatsAccuracy");

//        // 1 & 2 - choosing best lambda
//        // lambda 0 - 10
//        System.out.println("1&2a. Stats from NB all/pos features classifiers (wine, lambda 0 - 10):");
//        System.out.println("lambda,allFeatsAccuracy,posFeatsAccuracy");
//        for (double lambda = 0.0; lambda <= 10.0; lambda += 1.0) {
//            // set lambda
//            nbClassifierAllFeats.setLambda(lambda);
//            nbClassifierPosFeats.setLambda(lambda);
//
//            // get accuracies
//            double currAllFeatsAccuracy = trainTestClassifier(1, nbClassifierAllFeats, new ArrayList<>(), wineSplit);
//            double currPosFeatsAccuracy = trainTestClassifier(1, nbClassifierPosFeats, new ArrayList<>(), wineSplit);
//
//            // print csv stats
//            System.out.printf("%f,%f,%f\n", lambda, currAllFeatsAccuracy, currPosFeatsAccuracy);
//        }
//        System.out.println();
    }

    /**
     * Trains and tests a given classifier trained on the specified data set.
     * Data preprocessors can optionally be provided.
     * @param iterationCount number of iterations to average performance over
     * @param classifier which classifier to use
     * @param preprocessors a list of preprocessors in order of their intended use
     * @param dataSetSplit the data set to use
     * @return a double representing the accuracy of the test
     */
    public static double trainTestClassifier(
            int iterationCount,
            Classifier classifier,
            List<DataPreprocessor> preprocessors,
            DataSetSplit dataSetSplit) {

        // init accuracy stats
        int correctGuesses = 0;
        int totalGuesses = 0;

        // evaluate trained classifier
        for (int i = 1; i <= iterationCount; i++) {
            // split data
            DataSet trainData = dataSetSplit.getTrain();
            DataSet testData = dataSetSplit.getTest();

            // normalize data with specified preprocessors
            for (DataPreprocessor preprocessor : preprocessors) {
                preprocessor.preprocessTrain(trainData);
                preprocessor.preprocessTest(testData);
            }

            // train classifier
            classifier.train(trainData);

            for (Example ex : testData.getData()) {
                // classify
                double classification = classifier.classify(ex);

                // if correct, add to correct guesses
                if (classification == ex.getLabel()){
                    correctGuesses++;
                }

                // increment total guesses
                totalGuesses++;
            }
        }

        // return final test accuracy
        return (double)correctGuesses / (double)totalGuesses;
    }

    /**
     * Same as trainTestClassifier but incorporates stats from evaluating training data against the model.
     *
     * @param iterationCount number of iterations to average performance over
     * @param classifier which classifier to use
     * @param preprocessors a list of preprocessors in order of their intended use
     * @param dataSetSplit the data set to use
     */
    public static void trainTestClassifierWithTestAccuracy(
            int iterationCount,
            Classifier classifier,
            List<DataPreprocessor> preprocessors,
            DataSetSplit dataSetSplit) {

        // init accuracy stats
        int correctTrainGuesses = 0;
        int totalTrainGuesses = 0;
        int correctTestGuesses = 0;
        int totalTestGuesses = 0;

        // evaluate trained classifier
        for (int i = 1; i <= iterationCount; i++) {
            // split data
            DataSet trainData = dataSetSplit.getTrain();
            DataSet testData = dataSetSplit.getTest();

            // normalize data with specified preprocessors
            for (DataPreprocessor preprocessor : preprocessors) {
                preprocessor.preprocessTrain(trainData);
                preprocessor.preprocessTest(testData);
            }

            // train classifier
            classifier.train(trainData);

            // evaluate model accuracy over training data
            for (Example trainEx : trainData.getData()) {
                // classify
                double classification = classifier.classify(trainEx);

                // if correct, add to correct training guesses
                if (classification == trainEx.getLabel()){
                    correctTrainGuesses++;
                }

                // increment total training guesses
                totalTrainGuesses++;
            }

            // evaluate model accuracy over testing data
            for (Example testEx : testData.getData()) {
                // classify
                double classification = classifier.classify(testEx);

                // if correct, add to correct testing guesses
                if (classification == testEx.getLabel()){
                    correctTestGuesses++;
                }

                // increment total testing guesses
                totalTestGuesses++;
            }
        }
    }

    /**
     * Helper for printing model evaluation stats.
     *
     * @param correctGuesses
     * @param totalGuesses
     */
    public static void printStats(int correctGuesses, int totalGuesses) {
        System.out.printf("-- Correct guesses: %d\n", correctGuesses);
        System.out.printf("-- Total guesses: %d\n", totalGuesses);
        System.out.printf("-- Accuracy: %f%%\n", (double)correctGuesses / (double)totalGuesses);
        System.out.println("");
    }

    /**
     * Helper for printing model evaluation stats in a CSV-friendly way.
     *
     * @param iteration
     * @param correctGuesses
     * @param totalGuesses
     */
    public static void printCSVFriendlyStats(int iteration, int correctGuesses, int totalGuesses) {
        double accuracy = (double) correctGuesses / (double) totalGuesses;
        System.out.printf("%d, %f\n", iteration, accuracy);
    }

    /**
     * Helper for printing model evaluation stats (training and testing data).
     *
     * @param correctTrainGuesses
     * @param totalTrainGuesses
     * @param correctTestGuesses
     * @param totalTestGuesses
     */
    public static void printStatsWithTrainTestAccuracy(
            int correctTrainGuesses,
            int totalTrainGuesses,
            int correctTestGuesses,
            int totalTestGuesses) {
        System.out.printf("-- Correct training data guesses: %d\n", correctTrainGuesses);
        System.out.printf("-- Total training data guesses: %d\n", totalTrainGuesses);
        System.out.printf("-- Training data accuracy: %f%%\n", (double)correctTrainGuesses / (double)totalTrainGuesses);
        System.out.printf("-- Correct testing data guesses: %d\n", correctTestGuesses);
        System.out.printf("-- Total testing data guesses: %d\n", totalTestGuesses);
        System.out.printf("-- Testing data accuracy: %f%%\n", (double)correctTestGuesses / (double)totalTestGuesses);
        System.out.println("");
    }

    /**
     * Helper for printing model evaluation stats in a CSV-friendly way (training and testing stats).
     * Formatted iteration, trainAccuracy, testAccuracy
     *
     * @param iteration
     * @param correctTrainGuesses
     * @param totalTrainGuesses
     * @param correctTestGuesses
     * @param totalTestGuesses
     */
    public static void printCSVFriendlyStatsWithTrainAndTestAccuracy(
            int iteration,
            int correctTrainGuesses,
            int totalTrainGuesses,
            int correctTestGuesses,
            int totalTestGuesses) {
        double trainAccuracy = (double) correctTrainGuesses / (double) totalTrainGuesses;
        double testAccuracy = (double) correctTestGuesses / (double) totalTestGuesses;
        System.out.printf("%d, %f, %f\n", iteration, trainAccuracy, testAccuracy);
    }
}
