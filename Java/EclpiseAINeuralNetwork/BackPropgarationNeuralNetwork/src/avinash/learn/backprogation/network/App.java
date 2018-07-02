package avinash.learn.backprogation.network;

public class App {

	public static void main(String[] args) {

		float[][] trainingData = new float[][] {

				// YELLOW CIRCLES 1 -> (1,0,0)
				new float[] { 0.1f, 0.2f }, new float[] { 0.3f, 0.2f }, new float[] { 0.15f, 0.58f },
				new float[] { 0.45f, 0.7f }, new float[] { 0.4f, 0.9f },

				// GREEN CIRCLES 2 -> (0,1,0)
				new float[] { 0.4f, 1.2f }, new float[] { 0.45f, 0.95f }, new float[] { 0.42f, 1f },
				new float[] { 0.5f, 1.1f }, new float[] { 0.52f, 1.45f },

				// BLUE CIRCLES 3 -> (0,0,1)
				new float[] { 0.6f, 0.2f }, new float[] { 0.75f, 0.7f }, new float[] { 0.9f, 0.34f },
				new float[] { 0.85f, 0.76f }, new float[] { 0.8f, 0.34f } };

		float[][] trainingResults = new float[][] { new float[] { 1, 0, 0 }, new float[] { 1, 0, 0 },
				new float[] { 1, 0, 0 }, new float[] { 1, 0, 0 }, new float[] { 1, 0, 0 }, new float[] { 0, 1, 0 },
				new float[] { 0, 1, 0 }, new float[] { 0, 1, 0 }, new float[] { 0, 1, 0 }, new float[] { 0, 1, 0 },
				new float[] { 0, 0, 1 }, new float[] { 0, 0, 1 }, new float[] { 0, 0, 1 }, new float[] { 0, 0, 1 },
				new float[] { 0, 0, 1 } };

		BackPropgationNeuralNetwork backpropagationNeuralNetworks = new BackPropgationNeuralNetwork(2, 4, 3);

		// training
		for (int iterations = 0; iterations < NeuralNetConstants.ITERATION; iterations++) {
			for (int i = 0; i < trainingResults.length; i++) {
				backpropagationNeuralNetworks.train(trainingData[i], trainingResults[i],
						NeuralNetConstants.LEARNING_RATE, NeuralNetConstants.MOMENTUM);
			}
		}

		// testing
		float[] result = backpropagationNeuralNetworks.run(new float[] { 0.11f, 0.12f });
		System.out.print(result[0] + " - " + result[1] + " - " + result[2]);
	}

}
