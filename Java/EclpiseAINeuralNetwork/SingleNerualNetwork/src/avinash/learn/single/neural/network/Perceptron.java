package avinash.learn.single.neural.network;

public class Perceptron {
	private float[] weights;
	private float[][] input;
	private float[] output;
	private int numberOfWeights;
	
	public Perceptron(float[][] input, float[] output){
		this.input = input;
		this.output = output;
		this.numberOfWeights = input[0].length;
		this.weights = new float[numberOfWeights];
		initializeweight();

	}

	private void initializeweight() {
		// TODO Auto-generated method stub
		
	}
	
	public void train(float learningrate, int epocs){
		
	}
	
	public float calculateTheOutput(float[] output){
		
	}

}
