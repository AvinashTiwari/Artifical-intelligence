package avinash.learn.backprogation.network;

public class BackPropgationNeuralNetwork {
private Layer[] layers;
	
	public BackPropgationNeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
		layers = new Layer[2];
		layers[0] = new Layer(inputSize, hiddenSize);
		layers[1] = new Layer(hiddenSize, outputSize);
	}

	public Layer getLayer(int index) {
		return layers[index];
	}

	public float[] run(float[] input) {
		float[] inputActivation = input;
		for (int i = 0; i < layers.length; i++) {
			inputActivation = layers[i].run(inputActivation);
		}
		return inputActivation;
	}

	public void train(float[] input, float[] targetOutput, float learningRate, float momentum) {
		
		float[] calculatedOutput = run(input);
		float[] error = new float[calculatedOutput.length];
		
		for (int i = 0; i < error.length; i++) {
			error[i] = targetOutput[i] - calculatedOutput[i]; 
		}
		
		for (int i = layers.length - 1; i >= 0; i--) {
			error = layers[i].train(error, learningRate, momentum);
		}
	}

}
