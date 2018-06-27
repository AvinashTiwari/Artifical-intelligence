package avinash.learn.single.neural.network;

public class App {
 public static void main(String[] args) {
	 float[][] input = {{0,0} , {0,1}, {1,0} ,{1,1}};
	 float[] output = {0,0,0,1};

	 Perceptron perceptron = new Perceptron(input, output);
	 perceptron.train(0.01f, 200);
	 
	 System.out.println("There is error rate 0 our neural networ is ready");
	 
	 System.out.println(perceptron.calculateTheOutput(new float[]{0,0}));
	 System.out.println(perceptron.calculateTheOutput(new float[]{0,1}));
	 System.out.println(perceptron.calculateTheOutput(new float[]{1,0}));
	 System.out.println(perceptron.calculateTheOutput(new float[]{1,1}));




}
 
}
