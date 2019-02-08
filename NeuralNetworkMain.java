package net;

/* Neural network, with archetecture of VGG16, loaded from Models Zoo.
 * Pre-trained on dataset VGGFACE.
 * Training and tuning onle last layer.
 */

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NeuralNetworkMain {

	// Logger for this class
	public static Logger log = LoggerFactory.getLogger(NeuralNetworkMain.class);

	public static String fileName = new String("log.txt"); // »м€ файла дл€ сбора данных об обучении

	// Variable, that hold the place in file system, where params of net are stored
	// If collectStats = true, then net is not trained yet and need to be trained
	// If collectStats = false, then params are already saved in file and needed to be loaded
	public static boolean collectStats = false;

	//Where to save the network. Note: the file is in .zip format - can be opened externally
	public static File locationToSave = new File("EmotionsRecognitionNetwork.zip");

	//Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
	public static boolean saveUpdater = true;


	// Iterators for the data
	public static RecordReaderDataSetIterator trainIterator;
	public static RecordReaderDataSetIterator testIterator; // Iterator for test on dataset data
	public static RecordReaderDataSetIterator noiseTestIterator; // Iterator for test on data from the webcam with noise

	// Datasets data
	public static DataSet trainingData;
	public static DataSet testData;
	public static DataSet noiseTestData;

	//Images are of format given by allowedExtension
	private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

	public static int epochs = 3; // Number of epochs
	public static int[] inputShape = new int[] {3, 224, 224}; // Size of input image: cohn-kanade - [1x640x490] -> convert to VGGFACE - [3, 224, 224]
	public static int numLabels = 3; // Number of classes
	public static int labelIndex = 1; // For images always = 1
	public static long seed = 123; // Random seed
	public static Random randNumGen = new Random(seed); // Randomizer
	public static WorkspaceMode workspaceModeForTrain = WorkspaceMode.SEPARATE; // SEPARATE is slower, but used less memory. SINGLE is opposite.
	public static WorkspaceMode workspaceModeForInference = WorkspaceMode.SINGLE; // SEPARATE is slower, but used less memory. SINGLE is opposite.
	public static int batchSizeTraining = 5;
	public static int batchSizeTesting = 5;
	public static final String featureExtractionLayer = "fc7"; // Layer, under which all params are frozen



	public static void main(String[] args) throws  Exception {

		// Create file for saving config and results
		File file = new File(fileName);
		if(!file.exists()) {
			System.out.println("\n\nLog file does not exist. Create...");
			file.createNewFile();
			System.out.println("\n\nLog file created in folder: " + file.getAbsolutePath());
		}
		else {
			System.out.println("\n\nLog file created in folder: " + file.getAbsolutePath());
		}
		// Create FileWriter
		FileWriter writer = new FileWriter(file, false);

		writer.write("Build model....");
		writer.flush();
		System.out.println("Build model....");

		// Load Zoo model VGG16
		ZooModel zooModel = new VGG16();
		System.out.println("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
		ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.VGGFACE);

		writer.write("\n\nLoaded model overview: " + vgg16.summary());
		writer.flush();
		System.out.println("\n\nLoaded model overview: " + vgg16.summary());

		if(zooModel.pretrainedAvailable(PretrainedType.VGGFACE)) {
			System.out.println("VGGFACE coeffs are available");
		}

		// Create Fine tune configuration, which will modify params of all layers, that are not frozen
		FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
				.updater(new Nesterovs(5e-5))
				.seed(123)
				.build();

		/*
		 * Change to required config and print it.
		 * Therefore the last layer (as seen when printing the summary) is a dense layer and not an output layer with a loss function.
		 * Therefore to modify nOut of an output layer we delete the layer vertex,
		 * keeping itТs connections and add back in a new output layer with the same name,
		 * a different nOut, the suitable loss function etc etc.
		 * */
		ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
				.fineTuneConfiguration(fineTuneConf)
				.setFeatureExtractor(featureExtractionLayer)
				.removeVertexKeepConnections("fc8")
				.addLayer("fc8",
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.nIn(4096).nOut(numLabels)
								.weightInit(WeightInit.DISTRIBUTION)
								.dist(new NormalDistribution(0,0.2*(2.0/(4096 + numLabels)))) //This weight init dist gave better results than Xavier
								.activation(Activation.SOFTMAX).build()
						,"fc7").setOutputs("fc8")
				.build();

		writer.write("\n\nCorrected model overview: " + vgg16Transfer.summary());
		writer.flush();

		System.out.println(vgg16Transfer.summary());

		System.out.println("\n\nModel loaded from the Zoo.");

		writer.write("\n\nLoad data for training....");
		writer.flush();
		System.out.println("\n\nLoad data for training....");

		File parentDir = new ClassPathResource("resources").getFile();

		// Divide all data to training and testing datasets
		FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

		// Create ParentPathLabelGenerator, that will parse the parent dir and use the name of the subdirectories as label/class names
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

		// Create ParentPathLabelGenerator for noisy data, that will parse the parent dir and use the name of the subdirectories as label/class names
		ParentPathLabelGenerator noiseDataLlabelMaker = new ParentPathLabelGenerator();
		BalancedPathFilter noiseDataPathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, noiseDataLlabelMaker);

		// Split the image files into train and test. Specify the train test split as 80%,20%
		InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
		InputSplit trainDataInputSplit = filesInDirSplit[0];
		InputSplit testDataInputSplit = filesInDirSplit[1];

		/*
		 * Specifying a new record reader (one for testing on dataset, one for testing on real data, one for training)
		 * with the height and width you want the images to be resized to.
		 * Note that the images in this example are all of different size
		 * They will all be resized to the height and width specified below
		 * */
		ImageRecordReader trainRecordReader = new ImageRecordReader(inputShape[1],inputShape[2],inputShape[0],labelMaker);
		trainRecordReader.initialize(trainDataInputSplit);

		ImageRecordReader testRecordReader = new ImageRecordReader(inputShape[1],inputShape[2],inputShape[0], labelMaker);
		testRecordReader.initialize(testDataInputSplit);

		// ------------------------------------------ CREATE!
	        /*ImageRecordReader realTestRecordReader = new ImageRecordReader(inputShape[1],inputShape[2],inputShape[0], realDataLlabelMaker);
	        testRecordReader.initialize(realTestDataInputSplit);*/

		// Get the number of labels, ImageRecordReader founded in dir and check it
		int outputNum = trainRecordReader.numLabels();

		if(outputNum != numLabels)  {
			writer.write("Wrong number of clesses in database detected.");
			writer.flush();
			System.err.println("Wrong number of clesses in database detected.");
		}
		// Init iterators with datasets
		trainIterator = new RecordReaderDataSetIterator(trainRecordReader, batchSizeTraining, labelIndex, outputNum);
		trainIterator.setCollectMetaData(true);
		
		testIterator = new RecordReaderDataSetIterator(testRecordReader, batchSizeTesting, labelIndex, outputNum);
		testIterator.setCollectMetaData(true);
		//realTestIterator = new RecordReaderDataSetIterator(realTestData, batchSizeTesting, labelIndex, outputNum);
		//realTestIterator.setCollectMetaData(true);
		
		// Normaize input data: mean (Expected value) = 0; variance = 1
		NormalizerStandardize normalizer = new NormalizerStandardize();

		/*
		 * Check, if model is ready - then load it from file, then check it on test iterator.
		 * Else - train it and save params to the specified file.
		 */
		if(!collectStats) {
			writer.write("\n\n  Model is already pre-trained. Loading...");
			writer.flush();
			System.out.println("\n\n  Model is already pre-trained. Loading...");
			vgg16Transfer = load();

			// Get results and check them
			Evaluation eval = new Evaluation(numLabels);
			testIterator.reset();
			int mataDataCounter = 0;
			while(testIterator.hasNext()) {
				// If iterator has next dataset
				testData = testIterator.next();
				// Get meta-data from this dataset
				 @SuppressWarnings("rawtypes")
				List<?> exampleMetaData = testData.getExampleMetaData();
				Iterator<?> exampleMetaDataIterator = exampleMetaData.iterator();
				System.out.println("\n\n  Metadata from dataset #" + mataDataCounter + ":\n");
				while (exampleMetaDataIterator.hasNext()) {
					RecordMetaDataURI metaDataUri = (RecordMetaDataURI) exampleMetaDataIterator.next();
					System.out.println(metaDataUri.getLocation());
				}
				
				// Normalize data
				normalizer.fit(testData);
				INDArray labelMean = normalizer.getMean(); 
				INDArray labelStd = normalizer.getStd(); 
				System.out.println("\nMetrics of normalized test data: mean = " + labelMean + ", std = " + labelStd);
				writer.write("\nMetrics of normalized test data: mean = " + labelMean + ", std = " + labelStd);
				writer.flush();
				
				INDArray features = testData.getFeatures();
				INDArray labels = testData.getLabels();
				System.out.println("\n\n  Labels predicted #" + mataDataCounter + ":\n");
				
				int[][] labelsPredicted = labels.toIntMatrix();
				for (int i = 0; i < labelsPredicted.length; i++) {
					int[] predictedClass = new int[labelsPredicted.length];
					for (int j = 0; j < labelsPredicted[i].length; j++) {
						if(labelsPredicted[i][j] == 1) {
							predictedClass[i] = j;
						}
						System.out.print(predictedClass[i]);
					}
					System.out.println();
				}
				System.out.println(labels);
				INDArray predicted = vgg16Transfer.outputSingle(features);
				eval.eval(labels, predicted);
				
				mataDataCounter++;
				
			}

			writer.write("\n\n Cheack loaded model on test data...");
			writer.write(String.format("Evaluation on test data - [Accuracy: %.3f, P: %.3f, R: %.3f, F1: %.3f] ",
					eval.accuracy(), eval.precision(), eval.recall(), eval.f1()));
			writer.write(eval.stats());
			writer.write(eval.confusionToString());
			writer.flush();
			System.out.println("\n\n Cheack loaded model on test data...");
			System.out.println(String.format("Evaluation on test data - [Accuracy: %.3f, P: %.3f, R: %.3f, F1: %.3f] ",
					eval.accuracy(), eval.precision(), eval.recall(), eval.f1()));
			System.out.println(eval.stats());
			System.out.println(String.format("General false positive errors - %.3f \nFor each class - [%.3f, %.3f, %.3f]",
					eval.falsePositiveRate(), eval.falsePositiveRate(0), eval.falsePositiveRate(1), eval.falsePositiveRate(2)));
			System.out.println(String.format("General false negative errors - %.3f \nFor each class - [%.3f, %.3f, %.3f]",
					eval.falseNegativeRate(), eval.falseNegativeRate(0), eval.falseNegativeRate(1), eval.falseNegativeRate(2)));
			System.out.println(eval.confusionToString());
			
		}
		else {
			writer.write("\n\n  Model is not pre-trained.");
			writer.flush();
			System.out.println("\n\n  Model is not pre-trained.");

			// Set Listeners, to see results of evaluation during training process and testing
			vgg16Transfer.setListeners(new ScoreIterationListener(100));
//			Evaluation eval1 = new Evaluation(outputNum);
			writer.write("\n\n  Train model....");
			writer.flush();
			System.out.println("\n\n  Train model....");
			int iterationsCounter = 0;

			// Go through all data "epochs" - number of times
			for(int n = 0; n < epochs; n++)  {
				writer.write(String.format("Epoch %d started training", n + 1));
				writer.flush();
				System.out.println(String.format("Epoch %d started training", n + 1));

				//Reset training iterator to the new epoch
				trainIterator.reset();

				// Go through all data once, till it's end
				while (trainIterator.hasNext()) {

					iterationsCounter++;
					trainingData = trainIterator.next();
					normalizer.fit(trainingData);
					INDArray labelMean = normalizer.getMean(); 
					INDArray labelStd = normalizer.getStd(); 
					System.out.println("\nMetrics of normalized train data: mean = " + labelMean + ", std = " + labelStd);
					writer.write("\nMetrics of normalized train data: mean = " + labelMean + ", std = " + labelStd);
					writer.flush();
					vgg16Transfer.fit(trainingData);
					writer.write("*** Completed iteration number # " + iterationsCounter + "  ***");
					writer.flush();
					System.out.println("*** Completed iteration number # " + iterationsCounter + "  ***");

				}

				writer.write(String.format("Epoch %d finished training", n + 1));
				writer.flush();
				System.out.println(String.format("Epoch %d finished training", n + 1));

				// Get results and check them
				Evaluation eval = new Evaluation(numLabels);
				testIterator.reset();
				while(testIterator.hasNext()) {
					testData = testIterator.next();
					normalizer.fit(testData); 
					INDArray labelMean = normalizer.getMean(); 
					INDArray labelStd = normalizer.getStd(); 
					System.out.println("\nMetrics of normalized test data: mean = " + labelMean + ", std = " + labelStd);
					writer.write("\nMetrics of normalized test data: mean = " + labelMean + ", std = " + labelStd);
					writer.flush();
					INDArray features = testData.getFeatures();
					INDArray labels = testData.getLabels();
					INDArray predicted = vgg16Transfer.outputSingle(features);
					eval.eval(labels, predicted);
				}

				writer.write(String.format("Evaluation on test data - [Epoch %d] [Accuracy: %.3f, P: %.3f, R: %.3f, F1: %.3f] ",
						n + 1, eval.accuracy(), eval.precision(), eval.recall(), eval.f1()));
				writer.write(eval.stats());
				writer.flush();
				System.out.println(String.format("Evaluation on test data - [Epoch %d] [Accuracy: %.3f, P: %.3f, R: %.3f, F1: %.3f] ",
						n + 1, eval.accuracy(), eval.precision(), eval.recall(), eval.f1()));
				System.out.println(eval.stats());
				System.out.println(eval.confusionToString());

			}

			// Save model params
			writer.write("\n\n  Model is trained. Saving...");
			writer.flush();
			System.out.println("\n\n  Model is trained. Saving...");
			save(vgg16Transfer);
		}

	}


	// Save model to file
	public static void save(ComputationGraph net) throws Exception {
		ModelSerializer.writeModel(net, locationToSave, saveUpdater);
	}

	// Load model from file
	public static ComputationGraph load() throws Exception {
		ComputationGraph restored = ModelSerializer.restoreComputationGraph(locationToSave);
		return restored;
	}

}
