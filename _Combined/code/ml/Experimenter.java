package ml;

import ml.classifiers.Classifier;
import ml.classifiers.PerceptronClassifier;
import ml.classifiers.AveragePerceptronClassifier;

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

        // init classifiers
        PerceptronClassifier pClassifier = new PerceptronClassifier();
        AveragePerceptronClassifier apClassifier = new AveragePerceptronClassifier();


        // 1a. train and test perceptron classifier with max iteration of 10; split fraction = 0.8
        pClassifier.setIterations(10);
        trainTestClassifier("1a. Final stats from perceptron classifier (max iteration of 10 & 80/20 split over 100 iters.):", pClassifier, data, 0.8);

        // 1b. train and test avg perceptron classifier with max iteration of 10; split fraction = 0.8
        apClassifier.setIterations(10);
        trainTestClassifier("1b. Final stats from average perceptron classifier (max iteration of 10 & 80/20 split over 100 iters.):", apClassifier, data, 0.8);

        // 2a. train and test perceptron classifier with max iterations ranging from 0, 10, 20,..., 100; split fraction = 0.8
        for (int maxIters = 0; maxIters <= 20; maxIters += 2) {
            pClassifier.setIterations(maxIters);
            String expDescription = String.format("2a. Final stats from perceptron classifier (max iteration of %d & 80/20 split over 100 iters.):", maxIters);
            trainTestClassifier(expDescription, pClassifier, data, 0.8);
        }

        // 2b. train and test perceptron classifier with max iterations ranging from 0, 10, 20,..., 100; split fraction = 0.8
        for (int maxIters = 0; maxIters <= 20; maxIters += 2) {
            apClassifier.setIterations(maxIters);
            String expDescription = String.format("2b. Final stats from average perceptron classifier (max iteration of %d & 80/20 split over 100 iters.):", maxIters);
            trainTestClassifier(expDescription, apClassifier, data, 0.8);
        }
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
