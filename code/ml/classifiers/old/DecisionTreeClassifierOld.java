package ml.classifiers.old;

import ml.classifiers.Classifier;
import ml.data.DataSet;
import ml.data.Example;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * A class to represent a decision tree classifier
 *
 * Prepared for CS158 Assignment 02. Authored by David D'Attile
 */
public class DecisionTreeClassifierOld implements Classifier {
    private Integer depthLimit;
    private HashMap<Integer, String> featureMap;
    private DecisionTreeNodeOld root;

    /**
     * Initialize the decision tree classifier. At initialization, the
     * decision tree is a decision stump defaulting to a prediction of '0.0',
     * has no depth limit, and has no default dataSet.
     */
    public DecisionTreeClassifierOld(){
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
        // separate data
        ArrayList<Example> data = dataSet.getData();

        // assign feature map
        this.featureMap = dataSet.getFeatureMap();

        // parent majority label initialized as 0.0 at start of decision tree
        this.root = trainHelper(data, new HashSet<>(), 0, findMajorityLabel(0.0, data));
    }

    private DecisionTreeNodeOld trainHelper(
            ArrayList<Example> partitionedData,
            Set<Integer> usedFeatures,
            Integer currentDepth,
            Double parentMajorityLabel) {

        // find current majority label in data set
        double currentMajorityLabel = findMajorityLabel(parentMajorityLabel, partitionedData);

        // base case 1: all data belongs to same label -> pick that label
        if (labelsAreEqual(partitionedData)) {
            return new DecisionTreeNodeOld(currentMajorityLabel);
        }

        // base case 2: all data has same features -> pick majority label
        if (featuresAreEqual(partitionedData)) {
            return new DecisionTreeNodeOld(currentMajorityLabel);
        }

        // base case 3: out of features to examine -> pick majority label
        if (this.featureMap.keySet().size() == usedFeatures.size()) {
            return new DecisionTreeNodeOld(currentMajorityLabel);
        }

        // base case 4: no data left -> pick majority label of parent
        if (partitionedData.isEmpty()) {
            return new DecisionTreeNodeOld(parentMajorityLabel);
        }

        // base case 5: reached depth limit -> pick majority label
        if (this.depthLimit >= 0 && currentDepth.equals(this.depthLimit)) {
            return new DecisionTreeNodeOld(currentMajorityLabel);
        }

        // increment current decision tree depth
        int newDepth = currentDepth + 1;

        // TODO breaking ties???
        // calculate "score" for each feature
        // branch off best one; create new inner node
        int featureIndex = calculateBestFeatureIndex(usedFeatures, partitionedData);
        DecisionTreeNodeOld innerNode = new DecisionTreeNodeOld(featureIndex);

        // make copies of used features so that we're not updating same object!
        Set<Integer> refreshedUsedFeaturesLeft = new HashSet<>(usedFeatures);
        refreshedUsedFeaturesLeft.add(featureIndex);
        Set<Integer> refreshedUsedFeaturesRight = new HashSet<>(usedFeatures);
        refreshedUsedFeaturesRight.add(featureIndex);

        // create data_left and data_right
        ArrayList<ArrayList<Example>> splitData = splitDataByFeatureIndex(featureIndex, partitionedData);
        ArrayList<Example> dataLeft = splitData.get(0);
        ArrayList<Example> dataRight = splitData.get(1);

        // recurse! call trainHelper for L/R children
        innerNode.setLeft(trainHelper(dataLeft, refreshedUsedFeaturesLeft, newDepth, currentMajorityLabel));
        innerNode.setRight(trainHelper(dataRight, refreshedUsedFeaturesRight, newDepth, currentMajorityLabel));

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
        return classifyHelper(example, this.root);
    }

    /**
     * Recursive helper function for classifying an example on a learned tree.
     *
     * @param example
     * @param dtNode
     * @return A double value representing the predicted label.
     */
    private double classifyHelper(Example example, DecisionTreeNodeOld dtNode) {
        // check base case (i.e. if current node is a leaf)
        if (dtNode.isLeaf()) {
            return dtNode.prediction();
        }

        // get feature index internal node splits on
        int splitFeatureIndex = dtNode.getFeatureIndex();

        // get respective value at given feature index for the example
        double exampleFeatureIndexValue = example.getFeature(splitFeatureIndex);

        // recurse left or right
        if (exampleFeatureIndexValue == DecisionTreeNodeOld.LEFT_BRANCH) {
            return classifyHelper(example, dtNode.getLeft());
        } else {
            return classifyHelper(example, dtNode.getRight());
        }
    }

    /**
     * Not implemented!
     */
    @Override
    public double confidence(Example example) {
        return 0;
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
                    minTrainingError = currTrainingError;
                }
            }
        }

        return minTrainingErrorFeatureIndex;
    }

    /**
     * Determines the training error using supplied data for a given feature index.
     *
     * @param featureIndex
     * @param data
     * @return A double representing the training error (from 0.0 to 1.0) for the
     * given feature index.
     */
    private double calculateTrainingErrorForFeatureIndex(int featureIndex, ArrayList<Example> data) {
        // initialize variables
        int totalExamples = data.size();
        ArrayList<Example> binLeft = new ArrayList<>();
        ArrayList<Example> binRight = new ArrayList<>();

        // loop through and bin examples
        for (Example e : data) {
            if (e.getFeature(featureIndex) == DecisionTreeNodeOld.LEFT_BRANCH) {
                binLeft.add(e);
            } else {
                binRight.add(e);
            }
        }

        // calculate training error
        int binLeftMajorityCount = countMajorityLabel(binLeft);
        int binRightMajorityCount = countMajorityLabel(binRight);
        double accuracy = (double)(binLeftMajorityCount + binRightMajorityCount) / (double)totalExamples;

        // 1 - accuracy for training error
        return 1 - accuracy;
    }

    /**
     * Counts the number of examples in provided data that are members of the majority label.
     *
     * @param data
     * @return Integer representing count of majority label.
     */
    private int countMajorityLabel(ArrayList<Example> data) {
        // initialize counters
        int negLabel = 0;
        int posLabel = 0;

        // increment through data and count neg & pos features
        for (Example e : data) {
            if (e.getLabel() < 0) {
                negLabel++;
            } else {
                posLabel++;
            }
        }

        return Math.max(negLabel, posLabel);
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

            if (featureValue == DecisionTreeNodeOld.LEFT_BRANCH) {
                dataLeft.add(e);
            } else if (featureValue == DecisionTreeNodeOld.RIGHT_BRANCH) {
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
