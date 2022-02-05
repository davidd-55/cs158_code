package ml;

import ml.classifiers.Classifier;
import ml.classifiers.PerceptronClassifier;

/**
 * A class to run experiments for our classifier
 *
 * Prepared for CS158 Assignment 02. Authored by David D'Attile
 */
public class Experimenter {

    /**
     * Main method for running the experiments specified for Assignment 2.
     * Leverages the Random and DecisionTreeClassifier classifiers for experiments.
     * @param args
     */
    public static void main(String[] args) {
        // TODO: when done, port to A03 folder and run!
        // parse data with helper
        DataSet data = getData("/Users/daviddattile/Dev/cs158_code/_Combined/data/titanic-train.csv");


        // 1. train and test perceptron classifier with max iteration of 10; split fraction = 0.8
        PerceptronClassifier pClassifier = new PerceptronClassifier();
        pClassifier.setIterations(100);
        trainTestClassifier("1. Final stats from perceptron classifier (max iteration of 10 & 80/20 split over 100 iters.):", pClassifier, data, 0.8);
        System.out.println(pClassifier);


        /*
        // parse data with helper
        DataSet data = getData("/Users/daviddattile/Dev/cs158_code/Assignment02/data/titanic-train.csv");

        // 1. train and test random classifier
        RandomClassifier randClassifier = new RandomClassifier();
        trainTestClassifier("1. Final stats from random classifier (80/20 split over 100 iters.):", randClassifier, data, 0.8);

        // 2. train and test DT classifier with no depth limit; split fraction = 0.8
        DecisionTreeClassifier dtClassifier = new DecisionTreeClassifier();
        dtClassifier.setDepthLimit(-1);
        trainTestClassifier("2. Final stats from DT classifier (no depth limit & 80/20 split over 100 iters.):", dtClassifier, data, 0.8);

        // 3. train and test DT classifier with depth limit ranging from 0,1,2,...,10; split fraction = 0.8
        for (int depthLimit = 0; depthLimit <= 10; depthLimit++) {
            dtClassifier.setDepthLimit(depthLimit);

            String expDescription = String.format("3-%d. Final stats from DT classifier (depth limit of %d & 80/20 split over 100 iters.):", depthLimit, depthLimit);
            trainTestClassifier(expDescription, dtClassifier, data, 0.8);
        }

        // 4. train and test DT classifier (on train AND test data) with depth limit ranging from 0,1,2,...,10; split fraction = 0.8
        for (int depthLimit = 0; depthLimit <= 10; depthLimit++) {
            dtClassifier.setDepthLimit(depthLimit);

            String expDescription = String.format("4-%d. Final stats from DT classifier (depth limit of %d & 80/20 split over 100 iters.):", depthLimit, depthLimit);
            trainTestClassifierWithTestAccuracy(expDescription, dtClassifier, data, 0.8);
        }

        // 5. train and test DT classifier (on train AND test data) with no depth limit; split fraction ranging from 0.05,0.10,0.15,...,0.90
        dtClassifier.setDepthLimit(-1);
        int expNumber = 0;
        for (double splitFraction = 0.05; splitFraction < 0.91; splitFraction += 0.05) {
            String expDescription = String.format("5-%d. Final stats from DT classifier (no depth limit & %f train split over 100 iters.):", expNumber, splitFraction);
            trainTestClassifierWithTestAccuracy(expDescription, dtClassifier, data, splitFraction);
            expNumber++;

         */
    }

    /**
     * Trains, tests, and prints out evaluation statistics for a given classifier trained on the
     * specified data set. Training/testing data randomly split using the splitFraction parameter.
     *
     * @param expDescription
     * @param classifier
     * @param dataSet
     * @param splitFraction
     */
    public static void trainTestClassifier(String expDescription, Classifier classifier, DataSet dataSet, double splitFraction) {
        // init accuracy stats
        int correctGuesses = 0;
        int totalGuesses = 0;

        // evaluate trained classifier
        for (int i = 1; i <= 100; i++) {
            // split data
            DataSet[] dataSplit = dataSet.split(splitFraction);
            DataSet trainData = dataSplit[0];
            DataSet testData = dataSplit[1];

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
        System.out.println(expDescription);
        printStats(correctGuesses, totalGuesses);
    }

    /**
     * Same as trainTestClassifier but incorporates stats from evaluating training data against the model.
     *
     * @param expDescription
     * @param classifier
     * @param dataSet
     * @param splitFraction
     */
    public static void trainTestClassifierWithTestAccuracy(String expDescription, Classifier classifier, DataSet dataSet, double splitFraction) {
        // init accuracy stats
        int correctTrainGuesses = 0;
        int totalTrainGuesses = 0;
        int correctTestGuesses = 0;
        int totalTestGuesses = 0;

        // evaluate trained classifier
        for (int i = 1; i <= 100; i++) {
            // split data
            DataSet[] dataSplit = dataSet.split(splitFraction);
            DataSet trainData = dataSplit[0];
            DataSet testData = dataSplit[1];

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
        System.out.println(expDescription);
        printStatsWithTrainTestAccuracy(correctTrainGuesses, totalTrainGuesses, correctTestGuesses, totalTestGuesses);
    }

    /**
     * Helper for parsing the titanic data set.
     *
     * @return
     */
    public static DataSet getData(String fpath){
        return new DataSet(fpath);
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
}
