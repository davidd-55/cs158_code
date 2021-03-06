package ml.classifiers;

import java.util.*;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Gradient descent classifier allowing for two different loss functions and
 * three different regularization settings.
 * 
 * @author dkauchak
 * modified by David D'Attile for Assignment 06
 */
public class GradientDescentClassifier implements Classifier {
	// constants for the different surrogate loss functions
	public static final int EXPONENTIAL_LOSS = 0;
	public static final int HINGE_LOSS = 1;
	public static final int SQUARED_LOSS = 2;
	
	// constants for the different regularization parameters
	public static final int NO_REGULARIZATION = 0;
	public static final int L1_REGULARIZATION = 1;
	public static final int L2_REGULARIZATION = 2;
	public static final int L3_REGULARIZATION = 3;

	// hyperparameters
	private int loss;
	private int regularization;
	private double lambda;
	private double eta; // aka learning rate
	
	private HashMap<Integer, Double> weights; // the feature weights
	private double b; // the intersect weight
	
	protected int iterations;

	/**
	 * Initialize the Gradient Descent classifier. At initialization, the classifier
	 * uses the exponential loss fxn, no regularization, and has lambda/eta values of 0.01.
	 */
	public GradientDescentClassifier() {
		this.loss = EXPONENTIAL_LOSS;
		this.regularization = NO_REGULARIZATION;
		this.lambda = 0.01;
		this.eta = 0.01;
		this.weights = new HashMap<>();
		this.b = 0;
		this.iterations = 10;
	}

	/**
	 * Train this classifier based on the data set
	 *
	 * @param data
	 */
	@Override
	@SuppressWarnings("unchecked")
	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());

		ArrayList<Example> training = (ArrayList<Example>)data.getData().clone();

		for( int it = 0; it < this.iterations; it++ ){
			Collections.shuffle(training);
			double lossSum = 0.0; // for writeup questions 3 and 4

			for( Example e: training ) {
				double label = e.getLabel(); // y
				double prediction = getDistanceFromHyperplane(e, this.weights, this.b); // y'
				double c = calculateLoss(label, prediction); // loss(label, prediction)

				// update loss for printing
				lossSum += getLoss(label, prediction); // uncomment for reg calculation

				// update the weights
				//for( Integer featureIndex: weights.keySet() ){
				for( Integer featureIndex: e.getFeatureSet() ){

					// get grad desc
					double oldWeight = this.weights.get(featureIndex); // wj
					double featureValue = e.getFeature(featureIndex); // xij
					double r = calculateRegularization(oldWeight); // regularization(oldWeight)

					// update weights
					// wj = wj + eta ((yi * xij * c) - (lambda * r))
					double newWeight = oldWeight + this.eta * ((featureValue * label * c) - (this.lambda * r));
					weights.put(featureIndex, newWeight);
				}

				// regularization(bias)
				double rBias = calculateRegularization(this.b);

				// update bias on a per-example basis
				// b = b + eta ((yi * 1 * c) - (lambda * r))
				this.b += this.eta * ((label * c) - (this.lambda * rBias));

				// print weights for questions 1 and 2
				// System.out.printf("GD classifier weights for example %d at iteration %d\n", training.indexOf(e), it);
				// System.out.println(this + "\n");
			}

			// print loss summation per iteration for questions 3 and 4
			// printLoss(lossSum, it); // uncomment for loss printed at each iteration
		}
	}

	/**
	 * Classify the example.  Should only be called *after* train has been called.
	 *
	 * @param example
	 * @return the class label predicted by the classifier for this example
	 */
	@Override
	public double classify(Example example) {
		return getPrediction(example, this.weights, this.b);
	}
	
	@Override
	public double confidence(Example example) {
		return Math.abs(getDistanceFromHyperplane(example, this.weights, this.b));
	}

	/**
	 * Get the prediction from the current set of weights on this example
	 *
	 * @param e the example to predict
	 * @return
	 */
	private double getPrediction(Example e){
		return getPrediction(e, this.weights, this.b);
	}

	/**
	 * Get the prediction from the on this example from using weights w and inputB
	 *
	 * @param e example to predict
	 * @param w the set of weights to use
	 * @param inputB the b value to use
	 * @return the prediction
	 */
	private static double getPrediction(Example e, HashMap<Integer, Double> w, double inputB){
		double sum = getDistanceFromHyperplane(e,w,inputB);

		if( sum > 0 ){
			return 1.0;
		}else if( sum < 0 ){
			return -1.0;
		}else{
			return 0;
		}
	}

	/**
	 * Set the loss fxn gradient descent should use based on constants
	 *
	 * @param loss
	 */
	public void setLoss(int loss){
		switch (loss) {
			case EXPONENTIAL_LOSS -> this.loss = EXPONENTIAL_LOSS;
			case HINGE_LOSS -> this.loss = HINGE_LOSS;
			case SQUARED_LOSS -> this.loss = SQUARED_LOSS;
			default -> {
				String msg = String.format("expected valid loss specification, received %d", loss);
				throw new IllegalArgumentException(msg);
			}
		}
	}

	/**
	 * Set the regularization fxn gradient descent should use based on constants
	 *
	 * @param regularization
	 */
	public void setRegularization(int regularization){
		switch (regularization) {
			case NO_REGULARIZATION -> this.regularization = NO_REGULARIZATION;
			case L1_REGULARIZATION -> this.regularization = L1_REGULARIZATION;
			case L2_REGULARIZATION -> this.regularization = L2_REGULARIZATION;
			case L3_REGULARIZATION -> this.regularization = L3_REGULARIZATION;
			default -> {
				String msg = String.format("expected valid regularization specification, received %d", regularization);
				throw new IllegalArgumentException(msg);
			}
		}
	}

	/**
	 * Set the eta (learning rate) gradient descent should use
	 *
	 * @param eta
	 */
	public void setEta(double eta){
		this.eta = eta;
	}

	/**
	 * Set the lambda gradient descent should use
	 *
	 * @param lambda
	 */
	public void setLambda(double lambda){
		this.lambda = lambda;
	}

	/**
	 * Set the iterations gradient descent should use during training
	 *
	 * @param iterations
	 */
	public void setIterations(int iterations){
		this.iterations = iterations;
	}

	/**
	 * Calculate the loss based on the classifier's specified loss fxn
	 *
	 * @param label
	 * @param prediction
	 * @return a double representing the loss calculation
	 */
	private double calculateLoss(double label, double prediction) {
		return calculateLoss(label, prediction, this.loss);
	}

	/**
	 * A helper for calculating loss from a label, prediction, and loss fxn
	 *
	 * @param label
	 * @param prediction
	 * @param loss
	 * @return a double representing the specified loss calculation
	 */
	private static double calculateLoss(double label, double prediction, int loss) {

		// init loss calculation
		double lossCalc;

		// calc based on loss fxn
		switch (loss) {
			case HINGE_LOSS -> lossCalc = Math.max(0, 1 - (label * prediction));
			case EXPONENTIAL_LOSS -> lossCalc = Math.exp(-(label * prediction));
			case SQUARED_LOSS -> lossCalc = Math.pow(label - prediction, 2);
			default -> {
				String msg = String.format("expected valid loss specification for calculation, received %d", loss);
				throw new IllegalArgumentException(msg);
			}
		}

		return lossCalc;
	}

	/**
	 * Calculate the regularization based on the classifier's specified regularization fxn
	 *
	 * @param weight
	 * @return a double representing the regularization calculation
	 */
	private double calculateRegularization(double weight) {
		return calculateRegularization(weight, this.regularization);
	}

	/**
	 * A helper for calculating loss from a weight and regularization fxn
	 *
	 * @param weight
	 * @param regularization
	 * @return a double representing the specified loss calculation
	 */
	private static double calculateRegularization(double weight, int regularization) {

		// init regularization calculation
		double regCalc;

		// calc based on regularization fxn
		switch (regularization) {
			case NO_REGULARIZATION -> regCalc = 0;
			case L1_REGULARIZATION -> regCalc = weight >= 0 ? 1.0 : -1.0;
			case L2_REGULARIZATION -> regCalc = weight;
			case L3_REGULARIZATION -> regCalc = Math.pow(weight, 2);
			default -> {
				String msg = String.format("expected valid regularization specification for calculation, received %d", regularization);
				throw new IllegalArgumentException(msg);
			}
		}

		return regCalc;
	}

	/**
	 * A helper function for getting the loss of our classifier from a label and prediction.
	 *
	 * @param label
	 * @param prediction
	 * @return a double representing the loss
	 */
	private double getLoss(double label, double prediction) {

		double loss = calculateLoss(label, prediction);

		// loss = loss(y,y') + lambda * ||w||^
		if (this.regularization == L1_REGULARIZATION) {
			loss += this.lambda * calculatePNorm(this.weights.values(), 1);
		}

		// loss = loss(y,y') + lambda/2 * ||w||^2
		if (this.regularization == L2_REGULARIZATION) {
			loss += (this.lambda / 2.0) * calculatePNorm(this.weights.values(), 2);
		}

		return loss;
	}

	/**
	 * A helper function for calculating the p norm of a vector.
	 *
	 * @param vector
	 * @return A double representing the norm of a vector.
	 */
	private static double calculatePNorm(Collection<Double> vector, double p) {
		double normSum = 0.0;

		for (double d : vector) {
			normSum += Math.pow(d, p);
		}

		return Math.pow(normSum, 1.0 / p);
	}

	/**
	 * A helper function for determining an example's distance from the hyperplane
	 *
	 * @param e
	 * @param w
	 * @param inputB
	 * @return a double representing the distance from the hyperplane
	 */
	private static double getDistanceFromHyperplane(Example e, HashMap<Integer, Double> w, double inputB){
		double sum = inputB;
		
		//for(Integer featureIndex: w.keySet()){
		// only need to iterate over non-zero features
		for( Integer featureIndex: e.getFeatureSet()){
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}
		
		return sum;
	}

	/**
	 * Initialize the weights and the intersect value
	 *
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features){
		weights = getZeroWeights(features);
		b = 0;
	}

	/**
	 * Get a weight vector over the set of features with each weight
	 * set to 0
	 *
	 * @param features the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features){
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();

		for( Integer f: features){
			temp.put(f, 0.0);
		}

		return temp;
	}

	/**
	 * A helper fxn for printing out a supplied loss value alongside
	 * the current classifier's set hyperparameters
	 *
	 * @param lossVal
	 */
	private void printLoss(double lossVal, int iteration) {
		String hyperParamString;

		if (this.loss == HINGE_LOSS) {
			if (this.regularization == NO_REGULARIZATION) {
				hyperParamString = String.format("aggregate loss for iteration %d (hinge loss/no regularization): %f", iteration, lossVal);
			} else if (this.regularization == L1_REGULARIZATION) {
				hyperParamString = String.format("aggregate loss for iteration %d (hinge loss/L1 regularization): %f", iteration, lossVal);
			} else {
				hyperParamString = String.format("aggregate loss for iteration %d (hinge loss/L2 regularization): %f", iteration, lossVal);
			}
		} else {
			if (this.regularization == NO_REGULARIZATION) {
				hyperParamString = String.format("exp loss for iteration %d (hinge loss/no regularization): %f", iteration, lossVal);
			} else if (this.regularization == L1_REGULARIZATION) {
				hyperParamString = String.format("exp loss for iteration %d (hinge loss/L1 regularization): %f", iteration, lossVal);
			} else {
				hyperParamString = String.format("exp loss for iteration %d (hinge loss/L2 regularization): %f", iteration, lossVal);
			}
		}

		System.out.println(hyperParamString);
	}
	
	public String toString(){
		StringBuffer buffer = new StringBuffer();
		
		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);
		
		for(Integer index: temp){
			buffer.append(index + ":" + weights.get(index) + " ");
		}

		buffer.append("b:" + this.b + " ");
		
		return buffer.substring(0, buffer.length()-1);
	}
}
