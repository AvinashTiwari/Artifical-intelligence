package DeeplearningConvolutionMnist.DeeplearningConvolutionMnist;

import org.nd4j.linalg.activations.Activation;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class App {
	
   
    public static void main(String[] args) throws Exception {
        
    	//no RGB channels just a single channel
    	int numOfChannels = 1;
    	//digit classification: we have 10 classes (0,1,2...9)
        int numOfOutput = 10; 
        //64 items are processed at the same time
        int batchSize = 64; 
        //an epoch is defined as a full pass of the data set
        int numOfEpochs = 1;
        //the number of parameter updates in a row, for each minibatch
        int numOfIterations = 1;
        int seed = 123; 

        //create the training and test datasets
        DataSetIterator trainingDataset = new MnistDataSetIterator(batchSize,true,seed);
        DataSetIterator testDataset = new MnistDataSetIterator(batchSize,false,seed);

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(numOfIterations) 
                .regularization(true).l2(0.0005)            
                .learningRate(.01)             
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //Nesterov's momentum: keep track of the previous layer's gradient and use it as a way of updating the gradient
                .updater(Updater.NESTEROVS) 
                .list()
                //the filter (feature detector) size is 5x5
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //grayscale image so no RGB components just 1 channel
                        .nIn(numOfChannels)
                        //takes only a single step over and then down as it slides the filter across the input volume
                        .stride(1, 1)
                        //number of kernels: so the number of feature maps in the output
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                //pooling layer AKA subsampling layer with MAX pooling
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                		//filter size is 2x2
                        .kernelSize(2,2)
                        //it will step two values as it slides horizontally and two values when it steps down to the next row
                        .stride(2,2)
                        .build())
                //convolutional layers + pooling layers right after each other
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                //we use densly connected network
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numOfOutput)
                        .activation(Activation.SOFTMAX)
                        .build())
                //MNIST has 60k samples with 28x28x1 grayscale images
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .backprop(true).pretrain(false).build();   

        MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
        neuralNetwork.init();

        neuralNetwork.setListeners(new ScoreIterationListener(1));
        
        for( int i=0; i<numOfEpochs; i++ ) {       	
            neuralNetwork.fit(trainingDataset);
            Evaluation evaluation = neuralNetwork.evaluate(testDataset);
            System.out.println(evaluation.stats());
            testDataset.reset();
        }
    }
}
