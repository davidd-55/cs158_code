package ml;

import ml.classifiers.*;
import ml.data.*;
import ml.utils.HashMapCounter;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;

/**
 * A class to run Gradient Descent classifier experiments
 *
 * Prepared for CS158 Assignment 06. Authored by David D'Attile
 */
public class Experimenter {

    /**
     * Main method for running the experiments specified.
     * @param args
     */
    public static void main(String[] args) {

        // init datasets
        DataSet wineData = new DataSet("/Users/daviddattile/Dev/cs158_code/data/wines.train", DataSet.TEXTFILE);
        DataSetSplit wineSplit= wineData.split(0.8);

        // init classifier
        NBClassifier nbClassifier = new NBClassifier();
        nbClassifier.train(wineData);

        /*
        // 1 and 2. Argue for algorithmic correctness
        gdClassifier.setIterations(1);
        gdClassifier.setLoss(GradientDescentClassifier.HINGE_LOSS);
        gdClassifier.setRegularization(GradientDescentClassifier.L2_REGULARIZATION);
        gdClassifier.train(simpleData);

        // 3. GD train/test with hinge loss/no reg. on 80/20 split; binary titanic; eta/lambda = 0.01
        gdClassifier.setIterations(10);
        gdClassifier.setLoss(GradientDescentClassifier.HINGE_LOSS);
        gdClassifier.setRegularization(GradientDescentClassifier.NO_REGULARIZATION);
        String exp3Desc = "3. Train GD classifier with hinge loss/no reg. (80/20 split; binary titanic; eta/lambda = 0.01):";
        trainTestClassifier(exp3Desc, true, false, 1, 1, gdClassifier, new ArrayList<>(), titanicSplit);

         */
    }

    /**
     * Trains, tests, and prints out evaluation statistics for a given classifier trained on the
     * specified data set. Data preprocessors can optionally be provided.
     *
     * @param expDescription experiment description to be printed at each iteration
     * @param csvFriendly whether to print csv-friendly stats
     * @param currIteration provided for printing stats
     * @param iterationCount number of iterations to average performance over
     * @param classifier which classifier to use
     * @param preprocessors a list of preprocessors in order of their intended use
     * @param dataSetSplit the data set to use
     * @return a double representing the accuracy of the test
     */
    public static double trainTestClassifier(
            String expDescription,
            boolean printStats,
            boolean csvFriendly,
            int currIteration,
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

        // print final stats
        if (expDescription != null && !expDescription.isEmpty()) {
            System.out.println(expDescription);
        }

        if (printStats) {
            if (csvFriendly) {
                printCSVFriendlyStats(currIteration, correctGuesses, totalGuesses);
            } else {
                printStats(correctGuesses, totalGuesses);
            }
        }

        // return final test accuracy
        return (double)correctGuesses / (double)totalGuesses;
    }

    /**
     * Same as trainTestClassifier but incorporates stats from evaluating training data against the model.
     *
     * @param expDescription experiment description to be printed at each iteration
     * @param csvFriendly whether to print csv-friendly stats
     * @param currIteration provided for printing stats
     * @param iterationCount number of iterations to average performance over
     * @param classifier which classifier to use
     * @param preprocessors a list of preprocessors in order of their intended use
     * @param dataSetSplit the data set to use
     */
    public static void trainTestClassifierWithTestAccuracy(
            String expDescription,
            boolean csvFriendly,
            int currIteration,
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

        // print final stats
        if (expDescription != null && !expDescription.isEmpty()) {
            System.out.println(expDescription);
        }

        if (csvFriendly) {
            printCSVFriendlyStatsWithTrainAndTestAccuracy(
                    currIteration,
                    correctTrainGuesses, totalTrainGuesses,
                    correctTestGuesses, totalTestGuesses);
        } else {
            printStatsWithTrainTestAccuracy(
                    correctTrainGuesses, totalTrainGuesses,
                    correctTestGuesses, totalTestGuesses);
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
