package avinash.learn.hopefiled.network;

public class HopeFiledNetwork {

	private double[][] weightmatrix;

	public HopeFiledNetwork(int dimension) {
		this.weightmatrix = new double[dimension][dimension];
	}

	public void train(double[] pattern) {

		double[] patternPipolar = Utils.transform(pattern);
		double[][] patternMatrix = Matrix.createMatrix(patternPipolar.length, patternPipolar.length);
		patternMatrix = Matrix.outerProduct(patternPipolar);
		patternMatrix = Matrix.clearDigonal(patternMatrix);

		this.weightmatrix = Matrix.addMatrix(this.weightmatrix, patternMatrix);

	}

	public void recall(double[] pattern) {
		double[] patternBipolar = Utils.transform(pattern);
		double[] result = Matrix.matrixVectorMultiplication(this.weightmatrix, patternBipolar);
		for (int i = 0; i < patternBipolar.length; ++i) {
			result[i] = ActivationNetwork.stepFunction(result[i]);
		}

		for (int i = 0; i < patternBipolar.length; ++i) {
			if (patternBipolar[i] != result[i]) {
				System.out.println("Pattern is Not Recongnized");
				return;
			}
		}

		System.out.println("Pattern is Recongnized");

	}

}
