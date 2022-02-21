package ml.classifiers;

import ml.classifiers.old.DecisionTreeClassifierOld;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

public class ClassifierTimer {
	/**
	 * Calculates the time to train and test the classifier averaged over numRuns on
	 * 80/20 splits of the data
	 * 
	 * @param classifier
	 * @param dataset 
	 */
	public static void timeClassifier(Classifier classifier, DataSet dataset, int numRuns){
		long trainSum = 0;
		long classifySum = 0;
		
		for( int i = 0; i < numRuns; i++ ){
			DataSetSplit temp = dataset.split(0.8);
			DataSet train = temp.getTrain();
			DataSet test = temp.getTest();

			System.gc();
			long start = System.currentTimeMillis();
			classifier.train(train);
			trainSum += System.currentTimeMillis() - start;

			System.gc();
			start = System.currentTimeMillis();
			classifyExamples(classifier, test);
			classifySum += System.currentTimeMillis() - start;
		}

		System.out.println("Average train time: " + ((double)trainSum)/numRuns/1000 + "s");
		System.out.println("Average test time: " + ((double)classifySum)/numRuns/1000 + "s");
	}

	/**
	 * Classify all of the examples with the classifier. We don't care about the results
	 * just that the classify function gets called for all of the examples.
	 * 
	 * @param classifier
	 * @param dataset
	 */
	private static void classifyExamples(Classifier classifier, DataSet dataset){
		for( Example e: dataset.getData() ){
			classifier.classify(e);
		}
	}
	
	public static void main(String[] args){
		String csvFile = "/Users/daviddattile/Dev/cs158_code/data/wines.train";
		DataSet dataset = new DataSet(csvFile, DataSet.TEXTFILE);
		ClassifierFactory dtFactory = new ClassifierFactory(ClassifierFactory.DECISION_TREE, 3);
		ClassifierFactory avgPerceptronFactory = new ClassifierFactory(ClassifierFactory.PERCEPTRON, 10);

		int numRuns = 3;

		System.out.println("------------------------");
		System.out.println("OVA w/DT-3:");
		OVAClassifier ova = new OVAClassifier(dtFactory);
		timeClassifier(ova, dataset, numRuns);

		System.out.println("------------------------");
		System.out.println("AVA w/DT-3:");
		AVAClassifier ava = new AVAClassifier(dtFactory);
		timeClassifier(ava, dataset, numRuns);

		System.out.println("------------------------");
		System.out.println("OVA w/Avg. Perceptron-10:");
		OVAClassifier ovaPerc = new OVAClassifier(avgPerceptronFactory);
		timeClassifier(ovaPerc, dataset, numRuns);

		System.out.println("------------------------");
		System.out.println("AVA w/Avg Perceptron-10:");
		AVAClassifier avaPerc = new AVAClassifier(avgPerceptronFactory);
		timeClassifier(avaPerc, dataset, numRuns);
	}
}
