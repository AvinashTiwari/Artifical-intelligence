package avinash.learn.backprogation.network;

public class App {

	public static void main(String[] args) {
  
		 float[][] trainigData = new float[][]{
			 new float[] {0,0},
			 new float[] {0,1},
			 new float[] {1,0},
			 new float[] {1,1}


		 };
		 
		 float[][] trainingResult= new float[][]{
			new float[] {0},
			new float[] {0},
			new float[] {0},
			new float[] {1}
			
		 };
		
		 BackPropgationNeuralNetwork bpn = new BackPropgationNeuralNetwork(2, 3, 1);
		 for(int iteration =0 ; iteration< NeuralNetConstants.ITERATION; iteration++){
			 
			 for(int i=0; i < trainingResult.length;i++){
				 bpn.train(trainigData[i], trainingResult[i], NeuralNetConstants.LEARNING_RATE, NeuralNetConstants.MOMENTUM);
			 }
			 
			 
			 for(int i=0; i < trainingResult.length;i++){
				 float[] t = trainigData[i];
				 System.out.println("Number of Iteraton " + iteration +1);
				 System.out.println( t[0] +  t[1] + bpn.run(t)[0] );
			 }
			 
		 }
	}

}
