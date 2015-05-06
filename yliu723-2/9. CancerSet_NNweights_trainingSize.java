package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic
 * algorithm to find optimal weights to a neural network that is classifying
 * benign and malignant tumor.
 *
 */
public class CancerTest2 {
	private static Instance[] instances = initializeInstances();
	private static int inputLayer = 9, hiddenLayer = 12, outputLayer = 1,
			trainingIterations = 1000;
	private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

	private static ErrorMeasure measure = new SumOfSquaresError();

	private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
	private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

	private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
	private static String[] oaNames = { "RHC", "SA", "GA" };
	private static String results = "";

	private static DecimalFormat df = new DecimalFormat("0.000");
	

	public static void main(String[] args) {
		
		ArrayList<Double> accSize = new ArrayList<>();
		ArrayList<Integer> size = new ArrayList<>();
		// vary the training sample sizes
		for (int k = 2; k <=559; k *= 2) {
			Instance[] trainInstances = new Instance[k];
			Instance[] testInstances = new Instance[140];
			for (int i = 0; i < trainInstances.length; i++) {
				trainInstances[i] = instances[i];
			}
			for (int i = 0; i < 140; i++) {
				testInstances[i] = instances[instances.length - 1 - i];
			}
			DataSet trainset = new DataSet(trainInstances);

			for (int i = 0; i < oa.length; i++) {
				networks[i] = factory.createClassificationNetwork(new int[] {
						inputLayer, hiddenLayer, outputLayer });
				nnop[i] = new NeuralNetworkOptimizationProblem(trainset,
						networks[i], measure);//
			}

			oa[0] = new RandomizedHillClimbing(nnop[0]);
			oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
			oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

			for (int i = 0; i < oa.length; i++) {
				double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
				train(oa[i], networks[i], oaNames[i], trainInstances); 
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10, 9);

				Instance optimalInstance = oa[i].getOptimal();
				networks[i].setWeights(optimalInstance.getData());

				double predicted, actual;
				start = System.nanoTime();
				for (int j = 0; j < testInstances.length; j++) { 
					networks[i].setInputValues(instances[j].getData());
					networks[i].run();

					predicted = Double.parseDouble(instances[j].getLabel()
							.toString());
					actual = Double.parseDouble(networks[i].getOutputValues()
							.toString());

					double trash = Math.abs(predicted - actual) < 0.5 ? correct++
							: incorrect++;

				}
				end = System.nanoTime();
				testingTime = end - start;
				testingTime /= Math.pow(10, 9);

				results += "\ntrainingSize " + k + "\nResults for "
						+ oaNames[i] + ": \nCorrectly classified " + correct
						+ " instances."
						+ "\nIncorrectly classified "
						+ incorrect
						+ " instances.\nPercent correctly classified: "
						+ df.format(correct / (correct + incorrect) * 100)
						+ "%\nTraining time: "
						+ df.format(trainingTime) 
						+ " seconds\nTesting time: " + df.format(testingTime)
						+ " seconds\n";
				accSize.add(correct / (correct + incorrect) * 100);
				size.add(k);
			}
		}

		System.out.println(results);
		ArrayList<Double> accSize0 = new ArrayList<>();
		ArrayList<Double> accSize1 = new ArrayList<>();
		ArrayList<Double> accSize2 = new ArrayList<>();
		for (int i =0; i< accSize.size();i++){
			if (i%3==0) accSize0.add(accSize.get(i));
			if (i%3==1) accSize1.add(accSize.get(i));
			if (i%3==2) accSize2.add(accSize.get(i));
			
		}

		for (int i = 0; i < size.size(); i+=3) {
			System.out.println(size.get(i));
		}
		System.out.println("RHC: ");
		for (double value : accSize0) {
		    System.out.println(value);
		}
		System.out.println("SA: ");
		for (double value : accSize1) {
		    System.out.println(value);
		}
		System.out.println("GA: ");
		for (double value : accSize2) {
		    System.out.println(value);
		}
	}

	private static void train(OptimizationAlgorithm oa,
			BackPropagationNetwork network, String oaName,
			Instance[] trainInstances) {
		//System.out.println("\nError results for " + oaName + "\n---------------------------");

		for (int i = 0; i < trainingIterations; i++) {
			oa.train();

			double error = 0;
			for (int j = 0; j < trainInstances.length; j++) {
				network.setInputValues(trainInstances[j].getData());
				network.run();

				Instance output = instances[j].getLabel(), example = new Instance(
						network.getOutputValues());
				example.setLabel(new Instance(Double.parseDouble(network
						.getOutputValues().toString())));
				error += measure.value(output, example);
			}

			// System.out.println(df.format(error));
		}
	}

	private static Instance[] initializeInstances() {

		double[][][] attributes = new double[699][][];

		try {
			BufferedReader br = new BufferedReader(new FileReader(new File(
					"C:/Users/yancheng/Desktop/breast-cancer-wisconsin.data")));

			for (int i = 0; i < attributes.length; i++) {
				Scanner scan = new Scanner(br.readLine());
				scan.useDelimiter(",");

				attributes[i] = new double[2][];
				attributes[i][0] = new double[9]; // 9 attributes
				attributes[i][1] = new double[1];

				scan.next();// skip the first column: patient ID number
				for (int j = 0; j < 9; j++)
					attributes[i][0][j] = Double.parseDouble(scan.next());

				attributes[i][1][0] = Double.parseDouble(scan.next());
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		Instance[] instances = new Instance[attributes.length];

		for (int i = 0; i < instances.length; i++) {
			instances[i] = new Instance(attributes[i][0]);
			// diagnosis classifications 4 (malignant or 1) or 2 (benign or 0)
			instances[i]
					.setLabel(new Instance(attributes[i][1][0] < 3 ? 0 : 1));
		}

		// shuffle the instance array in a randomized order
		Random rgen = new Random();
		for (int i = 0; i < instances.length; i++) {
			int randomPosition = rgen.nextInt(instances.length);
			Instance temp = instances[i];
			instances[i] = instances[randomPosition];
			instances[randomPosition] = temp;
		}

		return instances;
	}
}
