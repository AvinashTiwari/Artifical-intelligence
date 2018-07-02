package avinash.learn.backprogation.network;

public class XoRApp {
public static void main(String[] args) throws Exception {

		float[][] trainingData = new float[][] { 
				new float[] { 0, 0 }, 
				new float[] { 0, 1 }, 
				new float[] { 1, 0 },
				new float[] { 1, 1 } 
		};

		float[][] trainingResults = new float[][] {
				new float[] { 0 }, 
				new float[] { 1 }, 
				new float[] { 1 },
				new float[] { 0 } 
		};

		BackPropgationNeuralNetwork backpropagationNeuralNetworks = new BackPropgationNeuralNetwork(2, 3,1);

		for (int iterations = 0; iterations < NeuralNetConstants.ITERATION; iterations++) {

			for (int i = 0; i < trainingResults.length; i++) {
				backpropagationNeuralNetworks.train(trainingData[i], trainingResults[i],
						NeuralNetConstants.LEARNING_RATE, NeuralNetConstants.MOMENTUM);
			}

			System.out.println();
			for (int i = 0; i < trainingResults.length; i++) {
				float[] t = trainingData[i];
				System.out.printf("%d epoch\n", iterations + 1);
				System.out.printf("%.1f, %.1f --> %.3f\n", t[0], t[1], backpropagationNeuralNetworks.run(t)[0]);
			}
		}
	}
}
