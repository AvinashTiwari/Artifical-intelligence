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
		for(int i=0; i< numberOfWeights; i++){
			weights[i] = 0;
		}
		
	}
	
	public void train(float learningrate, int epocs){
		float totalError = 1;
		while(totalError != 0 ){
			totalError = 0;
			for(int i=0; i < output.length; i++){
				float calcuatedoutput = calculateTheOutput(input[i]);
				float error = Math.abs(output[i] - calcuatedoutput);
				totalError = totalError +error;
				for(int j =0 ; j < numberOfWeights; j++){
					weights[j] = weights[j] +  learningrate * input[i][j] * error;
					System.out.println("Updated Weight :" + weights[j]);
				}
			}
		}
		
		 System.out.println("Keep on training the neural network " + totalError);
	}
	
	public float calculateTheOutput(float[] input){
		float sum = 0f;
		for(int i =0; i < input.length; ++i){
			sum = sum + weights[i] * input[i];
		}
		
		return ActivarionFunction.stepFunction(sum);
	}

}
