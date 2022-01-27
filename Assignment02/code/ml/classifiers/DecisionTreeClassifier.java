package ml.classifiers;

import ml.DataSet;
import ml.Example;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class DecisionTreeClassifier implements Classifier{
    private int depthLimit;
    private DataSet dataSet;
    private DecisionTreeNode root;

    /**
     * Initialize the decision tree classifier. At initialization, the
     * decision tree is a decision stump defaulting to a prediction of '0.0',
     * has no depth limit, and has no default dataSet.
     */
    public DecisionTreeClassifier(){
        this.depthLimit = -1;
        this.dataSet = null;
        this.root  = new DecisionTreeNode(0.0);
    }

    /**
     * Train this classifier based on the data set
     *
     * @param data
     */
    @Override
    public void train(DataSet data){

    }

    public DecisionTreeNode trainHelper(DataSet data, int currentDepth, double parentMajorityLabel){

        // parentMajorityLabel helpful for case #4!!

        return null;
    }

    /**
     * Classify the example.  Should only be called *after* train has been called.
     *
     * @param example
     * @return the class label predicted by the classifier for this example
     */
    @Override
    public double classify(Example example){
        return 0.0;
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
     * Determines if all labels in a data set are equivalent.
     * Expects at least 1 item in the data set.
     *
     * @param dataSet
     * @return True if all labels in the dataset are equivalent, false otherwise.
     */
    public boolean labelsAreEqual(DataSet dataSet){
        ArrayList<Example> data = dataSet.getData();
        double firstLabel = data.get(0).getLabel();

        return data.stream().allMatch(d -> d.getLabel() == firstLabel);
    }

    /**
     * Retrieves the majority label of the supplied
     *
     * @param dataSet
     * @return The majority label of the supplied data set.
     */
    public double findMajorityLabel(DataSet dataSet){
        ArrayList<Example> data = dataSet.getData();

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
     * Determines if all examples contain equivalent features in a data set.
     * Expects at least 1 item in the data set.
     *
     * @param dataSet
     * @return True if all example features in the dataset are equivalent, false otherwise.
     */
    public boolean featuresAreEqual(DataSet dataSet){
        ArrayList<Example> data = dataSet.getData();
        Example firstExample = data.get(0);

        return data.stream().allMatch(d -> d.equalFeatures(firstExample));
    }

    /**
     * Generates a string representation of the decision tree using the
     * DecisionTreeNode.treeString() methods.
     *
     * @return the string representation of the decision tree.
     */
    public String toString(){
        if (this.dataSet == null){
            return root.treeString();
        } else{
            return root.treeString(this.dataSet.getFeatureMap());
        }

    }
}
