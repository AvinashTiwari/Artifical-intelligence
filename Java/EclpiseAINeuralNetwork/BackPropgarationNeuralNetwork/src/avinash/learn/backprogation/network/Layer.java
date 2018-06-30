package avinash.learn.backprogation.network;

import java.util.Arrays;
import java.util.Random;

public class Layer {
	
	private float[] output;
	private float[] input;
	private float[] weights;
	private float[] dWeights;
	private Random random;
	
	public Layer(int inputSize, int outputSize){
		this.output = new float[outputSize];
		this.input = new float[inputSize + 1];
		this.weights = new float[(inputSize + 1) * outputSize];
		this.dWeights = new float[weights.length];
		this.random = new Random();
        initWeights();

	}

	private void initWeights() {
      for(int i =0; i < weights.length; i++){
    	  weights[i] = (random.nextFloat()-0.5f) * 4f;
      }		
      
	}
	
	public float[] run(float[] inputArray){
		System.arraycopy(inputArray, 0, input[0], 0, inputArray.length);
		input[input.length-1] = 1;
		int offset= 0;
		for(int i=0;i < output.length; i++){
			for(int j=0;i < input.length; j++){
				output[i] += weights[offset+j] * input[j];
			}
			output[i] =  AcivationFunction.sigmoid(output[i]);
			offset +=input.length;
		}
		
		return Arrays.copyOf(output, output.length);
	}
	
public float[] train(float[] error, float learningrate, float momentum){
		int offset =0;
		float[] nextError = new float[input.length];
		for(int i=0; i< output.length; i++){
			float delta = error[i] * AcivationFunction.dsigmoid(output[i]);
			
			for(int j=0; i < input.length; j++){
				int weightIndex = offset +j;
				nextError[j] = nextError[j] + weights[weightIndex] * delta;
				float dw = input[j] * delta * learningrate;
				weights[weightIndex ]+= dWeights[weightIndex] * momentum + dw;
				dWeights[weightIndex] =dw;
			}
			
			offset += input.length;
		}
		return nextError;
	}

}
