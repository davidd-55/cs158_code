package ml.classifiers;

import ml.DataSet;
import ml.Example;

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
    public void train(DataSet data){

    }

    /**
     * Classify the example.  Should only be called *after* train has been called.
     *
     * @param example
     * @return the class label predicted by the classifier for this example
     */
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
