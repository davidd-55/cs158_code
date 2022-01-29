package ml;

import ml.classifiers.Classifier;
import ml.classifiers.DecisionTreeClassifier;
import ml.classifiers.RandomClassifier;

/**
 * A class to run experiments for our classifier
 *
 * Prepared for CS158 Assignment 02. Authored by David D'Attile
 */
public class Experimenter {
    public static void main(String[] args) {
        // parse and split data with helper
        // DataSet[] dataSplit = getSplitTitanicData(0.8);
        DataSet[] dataSplit = getSplitDemoData(1);
        DataSet trainData = dataSplit[0];
        DataSet testData = dataSplit[1];

        // initialize random classifier
        // RandomClassifier randClassifier = new RandomClassifier();

        // initialize DT classifier
        DecisionTreeClassifier dtClassifier = new DecisionTreeClassifier();

        // set max DT depth
        dtClassifier.setDepthLimit(-1);

        // TODO: remove after testing :)
        dtClassifier.train(trainData);
        //System.out.println(dtClassifier);

        // train and test classifier
        // trainTestClassifier(dtClassifier, trainData, testData);
    }

    public static void trainTestClassifier(Classifier classifier, DataSet trainData, DataSet testData) {
        // train classifier
        classifier.train(trainData);

        // init accuracy stats
        int correctGuesses = 0;
        int totalGuesses = 0;

        // evaluate trained classifier
        for (int i = 1; i <= 100; i++) {
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

            if (i % 10 == 0) {
                // print stats
                System.out.printf("Stats after run %d:\n", i);
                printStats(correctGuesses, totalGuesses);
            }
        }

        // print final stats
        System.out.println("Final stats:");
        printStats(correctGuesses, totalGuesses);
    }

    public static DataSet[] getSplitTitanicData(double splitFraction){
        DataSet titanicData = new DataSet("/Users/daviddattile/Dev/cs158_code/Assignment02/data/titanic-train.csv");
        return titanicData.split(splitFraction);
    }

    public static DataSet[] getSplitDefaultData(double splitFraction){
        DataSet defaultData = new DataSet("/Users/daviddattile/Dev/cs158_code/Assignment02/data/default.csv");
        return defaultData.split(splitFraction);
    }

    public static DataSet[] getSplitDemoData(double splitFraction){
        DataSet demoData = new DataSet("/Users/daviddattile/Dev/cs158_code/Assignment02/data/demo.csv");
        return demoData.split(splitFraction);
    }

    public static void printStats(int correctGuesses, int totalGuesses) {
        System.out.printf("-- Correct guesses: %d\n", correctGuesses);
        System.out.printf("-- Total guesses: %d\n", totalGuesses);
        System.out.printf("-- Accuracy: %f%%\n", (double)correctGuesses / (double)totalGuesses);
        System.out.println("");
    }
}
