package avinash.learn.hopefiled.network;

public class App {

	public static void main(String[] args) {
		HopeFiledNetwork hp = new HopeFiledNetwork(4);
		hp.train(new double[] { 1, 0, 1, 0 });
		hp.train(new double[] { 1, 1, 1, 1 });
		
		
		hp.recall(new double[] { 1, 0, 1, 0 });


	}

}
