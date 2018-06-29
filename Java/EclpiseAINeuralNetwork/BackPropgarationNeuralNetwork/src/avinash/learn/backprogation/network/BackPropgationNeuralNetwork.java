package avinash.learn.backprogation.network;

public class BackPropgationNeuralNetwork {
	
	private Layer[] layers;
	
	public BackPropgationNeuralNetwork(int inputSize, int hiddenSize, int outputSize){
		layers = new Layer[2];
		layers[0] = new Layer(inputSize,hiddenSize);
		layers[1] = new Layer(inputSize,hiddenSize);

	}
	
	public Layer getLayer(int index){
		return layers[index];
	}

	public float[] run(float[] input){
		
		float[]  activation  = input;
		for(int i =0; i < layers.length; i++){
			activation = layers[i].run(activation);
		}
		
		return activation;
	}
	
	public void train(float[] input ,float targetOuput , float learningRate , float momemtum){
		float[] calculatedOutput = run(input);
		float[] error = new float[calculatedOutput.length];
		
		for(int i=0; i < error.length; i++){
			error[i]= targetOuput[i] - calculatedOutput[i];
		}
		
		for(int i=0; i < layers.length; i--){
			error[i]= layers[i].train(error, learningRate, momemtum);

		}

	}
}
