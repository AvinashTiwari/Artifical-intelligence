package DeeplearningXOR.DeeplearningXOR;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer.Builder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

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
		hiddenlayer.nOut(4);
		hiddenlayer.activation(Activation.SIGMOID);
		hiddenlayer.weightInit(WeightInit.DISTRIBUTION);
		hiddenlayer.dist(new UniformDistribution(0,1));
		
		DenseLayer.Builder hiddenlayer2 = new DenseLayer.Builder();
		hiddenlayer2.nIn(4);
		hiddenlayer2.nOut(4);
		hiddenlayer2.activation(Activation.SIGMOID);
		hiddenlayer2.weightInit(WeightInit.DISTRIBUTION);
		hiddenlayer2.dist(new UniformDistribution(0,1));

		
		
		Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
		outputLayerBuilder.nIn(4);
		outputLayerBuilder.nOut(2);
		
		outputLayerBuilder.activation(Activation.SOFTMAX);
		outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
		outputLayerBuilder.dist(new UniformDistribution(0,1));
		
		ListBuilder listBuilder = builder.list();
		listBuilder.layer(0, hiddenlayer.build());
	   listBuilder.layer(1, hiddenlayer2.build());
		listBuilder.layer(2, outputLayerBuilder.build());
		listBuilder.pretrain(false);
		
		MultiLayerConfiguration newtoerkConfiguration = listBuilder.build();
		MultiLayerNetwork neuralNetwork  = new MultiLayerNetwork(newtoerkConfiguration);
		neuralNetwork.init();
		neuralNetwork.setListeners(new ScoreIterationListener(100));
		neuralNetwork.fit(dataset);
		
		INDArray output = neuralNetwork.output(dataset.getFeatureMatrix());
		Evaluation eval = new Evaluation(2);
		eval.eval(dataset.getLabels(), output);
		
		System.out.println(eval.stats());
		
		System.out.println("New Predection");
		System.out.println(neuralNetwork.output(input));


	}
}
