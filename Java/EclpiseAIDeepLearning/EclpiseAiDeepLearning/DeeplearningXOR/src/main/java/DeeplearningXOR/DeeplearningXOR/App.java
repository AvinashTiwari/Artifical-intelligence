package DeeplearningXOR.DeeplearningXOR;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Hello world!
 *
 */
public class App {
	public static void main(String[] args) {
		INDArray input = Nd4j.zeros(4, 2);
		INDArray labels = Nd4j.zeros(4, 2);

		input.putScalar(new int[] { 0, 0 }, 0);
		input.putScalar(new int[] { 0, 1 }, 0);
		input.putScalar(new int[] { 1, 0 }, 1);
		input.putScalar(new int[] { 1, 1 }, 0);
		input.putScalar(new int[] { 2, 0 }, 0);
		input.putScalar(new int[] { 2, 1 }, 1);
		input.putScalar(new int[] { 3, 0 }, 1);
		input.putScalar(new int[] { 3, 1 }, 1);

		labels.putScalar(new int[] { 0, 0 }, 0);
		labels.putScalar(new int[] { 0, 1 }, 0);
		labels.putScalar(new int[] { 1, 0 }, 1);
		labels.putScalar(new int[] { 1, 1 }, 0);
		labels.putScalar(new int[] { 2, 0 }, 0);
		labels.putScalar(new int[] { 2, 1 }, 1);
		labels.putScalar(new int[] { 3, 0 }, 1);
		labels.putScalar(new int[] { 3, 1 }, 1);
		
		DataSet dataset = new DataSet(input, labels);
		
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.iterations(10000);
		builder.learningRate(0.1);
		builder.seed(3);
		builder.useDropConnect(false);
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.biasInit(0);
		builder.miniBatch(false);
		
		DenseLayer.Builder hiddenlayer = new DenseLayer.Builder();
		hiddenlayer.nIn(2);
		hiddenlayer.nOut(0);
		hiddenlayer.activation(Activation.SIGMOID);
		hiddenlayer.weightInit(WeightInit.DISTRIBUTION);
		hiddenlayer.dist(new UniformDistribution(0,1));

		
		
		

	}
}
