package ml.classifiers;

/**
 * A class to represent a two-layer NN classifier with the weights preset and the
 * number of hidden nodes fixed to 2.
 *
 * Prepared for CS158 Assignment 08. Authored by David D'Attile
 */
public class TwoLayerNNExample extends TwoLayerNN {
    /**
     * This class is the same as TwoLayerNN except with the weights initialized
     * to the values specified in the handout example and the number of hidden nodes
     * fixed to 2.
     */
    public TwoLayerNNExample() {
        super(2);
    }

    /**
     * A helper fxn for nicely printing the NN's weight values.
     */
    public void printWeights() {
        // print hidden weights
        System.out.println("Hidden weights: \n[ x1, x2, bias ]");
        for (double[] nodeWeights : this.hiddenWeights) {
            printArray(nodeWeights);
        }
        System.out.println();

        // print output weights
        System.out.println("Output weights: \n[ v1, v2, bias ]");
        printArray(this.outputWeights);
        System.out.println();
    }

    /**
     * A helper fxn for nicely printing the NN's node values.
     */
    public void printNodeValues() {
        // print pre-activation hidden nodes
        System.out.println("Hidden node vals: \n[ h1, h2, bias ]");
        printArray(this.hiddenLayerPostActivation);
        System.out.println();

        // print output weights
        System.out.println("Output node val:");
        System.out.println(this.outputPostActivation);
        System.out.println();
    }

    /**
     * A helper fxn for nicely printing a double array.
     *
     * @param array
     */
    private static void printArray(double[] array) {
        // init string builder
        StringBuilder s = new StringBuilder("[ ");

        // append values to string
        for (int i = 0; i < array.length; i++) {
            double val = array[i];

            // edge case for last val
            if (i < array.length - 1) {
                s.append(String.format("%f, ", val));
            } else {
                s.append(String.format("%f", val));
            }
        }

        System.out.println(s.append(" ]"));
    }

    /**
     * A helper fxn used to initialize the NN's weights to the example weights
     * provided in the handout.
     */
    protected void initializeWeights(int featureCount) {
        // [
        //  [ w11: -0.7, w21: 1.6, bias1: -1.8]
        //  [ w12: 0.03, w22: 0.6, bias2: -1.4]
        // ]
        this.hiddenWeights = new double[2][3];
        this.hiddenWeights[0][0] = -0.7;
        this.hiddenWeights[0][1] = 1.6;
        this.hiddenWeights[0][2] = -1.8;
        this.hiddenWeights[1][0] = 0.03;
        this.hiddenWeights[1][1] = 0.6;
        this.hiddenWeights[1][2] = -1.4;

        // [v1: -1.1, v2: -0.6, bias: 1.8]
        this.outputWeights = new double[3];
        this.outputWeights[0] = -1.1;
        this.outputWeights[1] = -0.6;
        this.outputWeights[2] = 1.8;
    }
}
