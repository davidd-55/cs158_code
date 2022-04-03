package ml;

import ml.classifiers.*;
import ml.data.*;

import java.util.ArrayList;
import java.util.List;

/**
 * A class to run two-layer NN classifier experiments
 *
 * Prepared for CS158 Assignment 08. Authored by David D'Attile
 */
public class Experimenter {

    /**
     * Main method for running the experiments specified.
     * @param args
     */
    public static void main(String[] args) {
        // init datasets demo, titanic, and titanic (sigmoid) data sets
        DataSet demoData = new DataSet("/Users/daviddattile/Dev/cs158_code/data/a08_demo.csv", DataSet.CSVFILE);

        DataSet titanicData = new DataSet("/Users/daviddattile/Dev/cs158_code/data/titanic-train.csv", DataSet.CSVFILE);
        DataSetSplit titanicSplit= titanicData.split(0.9);
        CrossValidationSet titanicXV = titanicData.getCrossValidationSet(10);

        DataSet titanicDataSigmoid = new DataSet("/Users/daviddattile/Dev/cs158_code/data/titanic-train.csv", DataSet.CSVFILE);
        changeLabel(titanicDataSigmoid, -1.0, 0.0);
        CrossValidationSet titanicSigmoidXV= titanicDataSigmoid.getCrossValidationSet(10);


        // Q1. Example network node and weight values
        System.out.println("Q1. Example network node and weight values:");

        // init network
        TwoLayerNNExample exampleNN = new TwoLayerNNExample();
        exampleNN.setEta(0.5);
        exampleNN.setIterations(1);

        // train and get weights
        exampleNN.train(demoData);
        exampleNN.printNodeValues();
        exampleNN.printWeights();

        // Q2. Visualize testing/training accuracy and loss during network training
        System.out.println("Q2. Visualize testing/training accuracy and loss during network training:");

        // init default network and train/print stats
        TwoLayerNN twoLayerNN = new TwoLayerNN(3);
        twoLayerNN.train(titanicSplit.getTrain(), titanicSplit.getTest());
        System.out.println();

        // Q3. Impact of number of hidden nodes on network accuracy
        System.out.println("Q3. Impact of number of hidden nodes on network accuracy:");
        for (int nodeCount = 1; nodeCount <= 10; nodeCount++) {
            // init NN with correct node count
            TwoLayerNN nn = new TwoLayerNN(nodeCount);

            double avgTrainSum = 0.0;
            double avgTestSum = 0.0;

            // loop through folds and print training stats
            System.out.printf("Node count: %d\n", nodeCount);
            System.out.println("fold,trainAccuracy,testAccuracy");
            for (int fold = 0; fold < 10; fold++) {
                // get data fold
                DataSetSplit titanicFold = titanicXV.getValidationSet(fold);

                // train and save accuracies
                double[] accuracies = trainTestClassifierWithTestAccuracy(1, nn, new ArrayList<>(), titanicFold);

                // tally avg sums and print
                avgTrainSum += accuracies[0];
                avgTestSum += accuracies[1];
                System.out.printf("%d,%f,%f\n", fold, accuracies[0], accuracies[1]);
            }

            System.out.printf("avg,%f,%f\n\n", avgTrainSum / 10, avgTestSum / 10);
        }

        // Q4. Find the ideal NN
        System.out.println("Q4. Find the ideal NN:");

        // init network; 7 hidden nodes from experiment 3
        TwoLayerNN idealNN = new TwoLayerNN(7);

        // eta 0 - 0.5; step of 0.01
        System.out.println("eta,avgTrainAccuracy,avgTestAccuracy");
        for (double eta = 0.0; eta <= 0.5; eta += 0.01) {
            // set eta
            idealNN.setEta(eta);

            // init avg train/test sums
            double avgTrainSum = 0.0;
            double avgTestSum = 0.0;

            for (int fold = 0; fold < 10; fold++) {
                // get data fold
                DataSetSplit titanicFold = titanicXV.getValidationSet(fold);

                // train and save accuracies
                double[] accuracies = trainTestClassifierWithTestAccuracy(1, idealNN, new ArrayList<>(), titanicFold);

                // tally avg sums
                avgTrainSum += accuracies[0];
                avgTestSum += accuracies[1];
            }

            System.out.printf("%f,%f,%f\n", eta, avgTrainSum / 10, avgTestSum / 10);
        }
        System.out.println();

        // iterations 1 - 200; step of 1
        idealNN.setEta(0.1); // reset to default
        System.out.println("iterations,avgTrainAccuracy,avgTestAccuracy");
        for (int i = 1; i <= 200; i++) {
            // set eta
            idealNN.setIterations(i);

            // init avg train/test sums
            double avgTrainSum = 0.0;
            double avgTestSum = 0.0;

            for (int fold = 0; fold < 10; fold++) {
                // get data fold
                DataSetSplit titanicFold = titanicXV.getValidationSet(fold);

                // train and save accuracies
                double[] accuracies = trainTestClassifierWithTestAccuracy(1, idealNN, new ArrayList<>(), titanicFold);

                // tally avg sums
                avgTrainSum += accuracies[0];
                avgTestSum += accuracies[1];
            }

            System.out.printf("%d,%f,%f\n", i, avgTrainSum / 10, avgTestSum / 10);
        }
        System.out.println();

        // print 10-XV accuracies of ideal NN
        // set to found ideal values
        idealNN.setEta(0.39);
        idealNN.setIterations(144);

        // init avg train/test sums
        double avgTrainSum = 0.0;
        double avgTestSum = 0.0;

        System.out.println("fold,avgTrainAccuracy,avgTestAccuracy");
        for (int fold = 0; fold < 10; fold++) {
            // get data fold
            DataSetSplit titanicFold = titanicXV.getValidationSet(fold);

            // train and save accuracies
            double[] accuracies = trainTestClassifierWithTestAccuracy(1, idealNN, new ArrayList<>(), titanicFold);

            // tally avg sums
            avgTrainSum += accuracies[0];
            avgTestSum += accuracies[1];
            System.out.printf("%d,%f,%f\n", fold, accuracies[0], accuracies[1]);
        }

        System.out.printf("avg,%f,%f\n", avgTrainSum / 10, avgTestSum / 10);

        // Q5. Evaluate tanh vs. sigmoid performance (7 hidden nodes)
        System.out.println("Q5. Evaluate tanh vs. sigmoid performance (7 hidden nodes):");

        // init networks with default eta/iterations
        TwoLayerNN tanhNetwork = new TwoLayerNN(7);
        tanhNetwork.setTanhActivation();
        TwoLayerNN sigmoidNetwork = new TwoLayerNN(7);
        sigmoidNetwork.setSigmoidActivation();

        // init avg train/test sums
        double avgTanhTrainSum = 0.0;
        double avgTanhTestSum = 0.0;
        double avgSigmoidTrainSum = 0.0;
        double avgSigmoidTestSum = 0.0;

        System.out.println("fold,avgTanhTrainAccuracy,avgTanhTestAccuracy,avgSigmoidTrainAccuracy,avgSigmoidTestAccuracy");
        for (int fold = 0; fold < 10; fold++) {
            // get data folds (preprocessed sigmoid data)
            DataSetSplit titanicTanhFold = titanicXV.getValidationSet(fold);
            DataSetSplit titanicSigmoidFold = titanicSigmoidXV.getValidationSet(fold);

            // train and save accuracies
            double[] tanhAccuracies = trainTestClassifierWithTestAccuracy(1, tanhNetwork, new ArrayList<>(), titanicTanhFold);
            double[] sigmoidAccuracies = trainTestClassifierWithTestAccuracy(1, sigmoidNetwork, new ArrayList<>(), titanicSigmoidFold);

            // tally avg sums
            avgTanhTrainSum += tanhAccuracies[0];
            avgTanhTestSum += tanhAccuracies[1];
            avgSigmoidTrainSum += sigmoidAccuracies[0];
            avgSigmoidTestSum += sigmoidAccuracies[1];

            // print fold accuracies
            System.out.printf("%d,%f,%f,%f,%f\n", fold,
                    tanhAccuracies[0], tanhAccuracies[1],
                    sigmoidAccuracies[0], sigmoidAccuracies[1]);
        }

        System.out.printf("avg,%f,%f,%f,%f\n",
                avgTanhTrainSum / 10, avgTanhTestSum / 10,
                avgSigmoidTrainSum / 10, avgSigmoidTestSum / 10);
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
     *
     * @return an ArrayList where the item at index 0 is the training accuracy and the item
     * at index 1 is the testing accuracy
     */
    public static double[] trainTestClassifierWithTestAccuracy(
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

        // calc and return train/test accuracy
        double trainAccuracy = (double)correctTrainGuesses / (double)totalTrainGuesses;
        double testAccuracy = (double)correctTestGuesses / (double)totalTestGuesses;

        double[] accuracies = new double[2];
        accuracies[0] = trainAccuracy;
        accuracies[1] = testAccuracy;

        return accuracies;
    }

    /**
     * A small helper fxn for changing labels in a data set.
     *
     * @param data
     * @param label
     * @param target
     */
    private static void changeLabel(DataSet data, double label, double target) {
        // loop through examples and change label if necessary
        for (Example e : data.getData()) {
            if (e.getLabel() == label) {
                e.setLabel(target);
            }
        }
    }
}
