package ml.classifiers;

import ml.DataSet;
import ml.Example;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * A class to represent a decision tree classifier
 *
 * Prepared for CS158 Assignment 02. Authored by David D'Attile
 */
public class DecisionTreeClassifier implements Classifier{
    private Integer depthLimit;
    private HashMap<Integer, String> featureMap;
    private DecisionTreeNode root;

    /**
     * Initialize the decision tree classifier. At initialization, the
     * decision tree is a decision stump defaulting to a prediction of '0.0',
     * has no depth limit, and has no default dataSet.
     */
    public DecisionTreeClassifier(){
        this.depthLimit = -1;
        this.featureMap = null;
        this.root = null;
    }

    /**
     * Train this classifier based on the data set. If training data set is empty,
     * examples will be classified with label '0.0'.
     *
     * @param dataSet training data
     */
    @Override
    public void train(DataSet dataSet){
        ArrayList<Example> data = dataSet.getData();

        System.out.printf("Data labels are equal: %b\n", labelsAreEqual(data));
        System.out.printf("Data features are equal: %b\n", featuresAreEqual(data));
        System.out.printf("Majority label: %f\n", findMajorityLabel(0.0, data));
        System.out.printf("Data has majority label: %b\n", majorityLabelExists(data));

        this.featureMap = dataSet.getFeatureMap();

        // parent majority label initialized as 0.0 at start of decision tree
        this.root = trainHelper(data, new HashSet<Integer>(), 0, findMajorityLabel(0.0, data));
    }

    private DecisionTreeNode trainHelper(
            ArrayList<Example> partitionedData,
            Set<Integer> usedFeatures,
            Integer currentDepth,
            Double parentMajorityLabel) {

        // find current majority label in data set
        double currentMajorityLabel = findMajorityLabel(parentMajorityLabel, partitionedData);

        // base case 1: all data belongs to same label -> pick that label
        if (labelsAreEqual(partitionedData)) {
            return new DecisionTreeNode(currentMajorityLabel);
        }

        // base case 2: all data has same features -> pick majority label
        if (featuresAreEqual(partitionedData)) {
            return new DecisionTreeNode(currentMajorityLabel);
        }

        // base case 3: out of features to examine -> pick majority label
        if (this.featureMap.keySet().size() == usedFeatures.size()) {
            return new DecisionTreeNode(currentMajorityLabel);
        }

        // base case 4: no data left -> pick majority label of parent
        if (partitionedData.isEmpty()) {
            return new DecisionTreeNode(parentMajorityLabel);
        }

        // base case 5: reached depth limit -> pick majority label
        if (this.depthLimit >= 0 && currentDepth.equals(this.depthLimit)) {
            return new DecisionTreeNode(currentMajorityLabel);
        }

        // increment current decision tree depth
        int newDepth = currentDepth + 1;

        // calculate "score" for each feature
        // branch off best one; create new inner node
        int featureIndex = calculateBestFeatureIndex(usedFeatures, partitionedData);
        DecisionTreeNode innerNode = new DecisionTreeNode(featureIndex);

        // make copies of used features so that we're not updating same object!
        Set<Integer> refreshedUsedFeatures = new HashSet<>(usedFeatures);
        refreshedUsedFeatures.add(featureIndex);

        // create data_left and data_right
        ArrayList<ArrayList<Example>> splitData = splitDataByFeatureIndex(featureIndex, partitionedData);
        ArrayList<Example> dataLeft = splitData.get(0);
        ArrayList<Example> dataRight = splitData.get(1);

        // recurse! call trainHelper for L/R children
        innerNode.setLeft(trainHelper(dataLeft, refreshedUsedFeatures, newDepth, currentMajorityLabel));
        innerNode.setRight(trainHelper(dataRight, refreshedUsedFeatures, newDepth, currentMajorityLabel));

        return innerNode;
    }

    /**
     * Classify the example.  Should only be called *after* train has been called.
     *
     * @param example an example from a test data set
     * @return the class label predicted by the classifier for this example
     */
    @Override
    public double classify(Example example){
        return 0.0;
    }

    /**
     * Determines if all labels in a data set are equivalent.
     *
     * @param data
     * @return True if all labels in the dataset are equivalent,
     * false otherwise or if data set is empty.
     */
    private static boolean labelsAreEqual(ArrayList<Example> data) {
        // empty data set check
        if (data.isEmpty()) {
            return false;
        }

        double firstLabel = data.get(0).getLabel();

        return data.stream().allMatch(d -> d.getLabel() == firstLabel);
    }

    /**
     * Determines if all examples contain equivalent features in a data set.
     *
     * @param data
     * @return True if all example features in the dataset are equivalent,
     * false otherwise or if data set is empty.
     */
    private static boolean featuresAreEqual(ArrayList<Example> data) {
        // empty data set check
        if (data.isEmpty()) {
            return false;
        }

        Example firstExample = data.get(0);

        return data.stream().allMatch(d -> d.equalFeatures(firstExample));
    }

    /**
     * Retrieves the majority label in a data set.
     *
     * @param data
     * @return The majority label of the supplied data set.
     */
    private static double findMajorityLabel(
            Double parentMajorityLabel,
            ArrayList<Example> data) {

        // if no majority label, return parent majority label
        if (!majorityLabelExists(data)) {
            return parentMajorityLabel;
        }

        // this is a general solution that will work for data with labels that
        // aren't just binary!
        return data.stream()
                .map(Example::getLabel)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
                .entrySet()
                .stream()
                .max(Map.Entry.comparingByValue())
                .orElseThrow()
                .getKey();
    }

    /**
     * Determines if the data set contains a majority label.
     *
     * @param data
     * @return True if the data set has a majority label,
     * false otherwise or if data set is empty.
     */
    private static boolean majorityLabelExists(ArrayList<Example> data) {
        // empty data set check
        if (data.isEmpty()) {
            return false;
        }

        // edge case for all data have same labels
        if (labelsAreEqual(data)) {
            return true;
        }

        long uniqueLabelCounts = data.stream()
                .map(Example::getLabel)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
                .values()
                .stream()
                .distinct()
                .count();

        return uniqueLabelCounts != 1;
    }

    /**
     * Determines the best feature index for splitting a data set of examples based
     * on each feature's training error.
     *
     * @param usedFeatures
     * @param data
     * @return An integer representing the best feature index for splitting the provided
     * data on.
     */
    private int calculateBestFeatureIndex(Set<Integer> usedFeatures, ArrayList<Example> data) {
        // declare starter vars
        int minTrainingErrorFeatureIndex = Integer.MAX_VALUE;
        double minTrainingError = Double.MAX_VALUE;

        // loop through all feature indices
        for (int featureIndex : this.featureMap.keySet()) {
            // check if feature index has already been used
            if (!usedFeatures.contains(featureIndex)) {
                // calculate training error for current feature index
                double currTrainingError = calculateTrainingErrorForFeatureIndex(featureIndex, data);

                // reassign feature index representing min training error if appropriate
                if (currTrainingError < minTrainingError) {
                    minTrainingErrorFeatureIndex = featureIndex;
                }
            }
        }

        return minTrainingErrorFeatureIndex;
    }

    // TODO: implement
    /**
     * Determines the training error using supplied data for a given feature index.
     *
     * @param featureIndex
     * @param data
     * @return A double representing the training error (from 0.0 to 1.0) for the
     * given feature index.
     */
    private double calculateTrainingErrorForFeatureIndex(int featureIndex, ArrayList<Example> data) {
        return 0.0;
    }

    /**
     * Splits a list of examples from a data set by feature values for a specified featureIndex.
     *
     * @param featureIndex
     * @param data
     * @return An array of size two containing two ArrayLists of examples.
     * The ArrayList at index 0 contains examples where the feature is 'DecisionTreeNode.LEFT_BRANCH' at featureIndex,
     * and the ArrayList at index 1 contains examples where the feature is 'DecisionTreeNode.RIGHT_BRANCH' at featureIndex.
     */
    private ArrayList<ArrayList<Example>> splitDataByFeatureIndex(int featureIndex, ArrayList<Example> data) {
        // initialize ArrayLists
        ArrayList<Example> dataLeft = new ArrayList<>();
        ArrayList<Example> dataRight = new ArrayList<>();

        // loop through data
        for (Example e : data) {
            double featureValue = e.getFeature(featureIndex);

            if (featureValue == DecisionTreeNode.LEFT_BRANCH) {
                dataLeft.add(e);
            } else if (featureValue == DecisionTreeNode.RIGHT_BRANCH) {
                dataRight.add(e);
            }else {
                String message = String.format("Data contains an invalid feature label for feature at index %d with value %f.", featureIndex, featureValue);
                throw new IllegalArgumentException(message);
            }
        }

        // create return ArrayList
        ArrayList<ArrayList<Example>> splitData = new ArrayList<>();
        splitData.add(dataLeft);
        splitData.add(dataRight);

        return splitData;
    }

    /**
     * Sets the limit of the depth of the tree that will be generated when training.
     *
     * @param depth set to '-1' to learn the entire tree, '0' for a decision stump,
     *              '1' for an internal node with two leaf nodes, etc.
     *              Initialized as '-1'.
     */
    public void setDepthLimit(int depth){
        this.depthLimit = depth;
    }

    /**
     * Generates a string representation of the decision tree using the
     * DecisionTreeNode.treeString() methods.
     *
     * @return the string representation of the decision tree.
     */
    public String toString() {
        return this.root.treeString(this.featureMap);
    }
}
