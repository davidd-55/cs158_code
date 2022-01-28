package ml.classifiers;

import ml.DataSet;
import ml.Example;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

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
        System.out.printf("Data labels are equal: %b\n", labelsAreEqual(dataSet));
        System.out.printf("Data features are equal: %b\n", featuresAreEqual(dataSet));
        System.out.printf("Majority label: %f\n", findMajorityLabel(0.0, dataSet));
        System.out.printf("Data has majority label: %b\n", majorityLabelExists(dataSet));

        //this.featureMap = dataSet.getFeatureMap();

        // parent majority label
        //this.root = trainHelper(dataSet, new HashSet<Integer>(), 0, findMajorityLabel(0.0, dataSet));
    }

    private DecisionTreeNode trainHelper(
            DataSet partitionedData,
            HashSet<Integer> usedFeatures,
            Integer currentDepth,
            Double parentMajorityLabel) {

        Double currentMajorityLabel = findMajorityLabel(parentMajorityLabel, partitionedData);

        // base case 1: all data belongs to same label -> pick that label
        if (labelsAreEqual(partitionedData)) {
            return new DecisionTreeNode(currentMajorityLabel);
        }

        // base case 2: all data has same features -> pick majority label
        if (featuresAreEqual(partitionedData)) {
            return new DecisionTreeNode(currentMajorityLabel);
        }

        // TODO: implement base case check logic
        // base case 3: out of features to examine -> pick majority label
        if (false) {

        }

        // base case 4: no data left -> pick majority label of parent
        if (partitionedData.getData().isEmpty()) {
            return new DecisionTreeNode(parentMajorityLabel);
        }

        // base case 5: reached depth limit -> pick majority label
        if (this.depthLimit >= 0 && currentDepth.equals(this.depthLimit)) {
            return new DecisionTreeNode(currentMajorityLabel);
        }


        // make copies of used features so that we're not updating same object!

        return null;
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
     * @param dataSet
     * @return True if all labels in the dataset are equivalent,
     * false otherwise or if data set is empty.
     */
    private static boolean labelsAreEqual(DataSet dataSet) {
        ArrayList<Example> data = dataSet.getData();

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
     * @param dataSet
     * @return True if all example features in the dataset are equivalent,
     * false otherwise or if data set is empty.
     */
    private static boolean featuresAreEqual(DataSet dataSet) {
        ArrayList<Example> data = dataSet.getData();

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
     * @param dataSet a non-empty data set
     * @return The majority label of the supplied data set.
     */
    private static double findMajorityLabel(
            Double parentMajorityLabel,
            DataSet dataSet) {

        ArrayList<Example> data = dataSet.getData();

        // if no majority label, return parent majority label
        if (!majorityLabelExists(dataSet)) {
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
     * @param dataSet
     * @return True if the data set has a majority label,
     * false otherwise or if data set is empty.
     */
    private static boolean majorityLabelExists(DataSet dataSet) {
        ArrayList<Example> data = dataSet.getData();

        // empty data set check
        if (data.isEmpty()) {
            return false;
        }

        // edge case for all data have same labels
        if (labelsAreEqual(dataSet)) {
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
