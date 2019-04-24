package application;


import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.stage.Stage;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.ScrollPane.ScrollBarPolicy;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.geometry.HPos;
import javafx.geometry.Insets;
import javafx.geometry.Pos;

public class Main extends Application {

    // FXML variables

    @FXML
    private Button stat_charts; // Button to show other statistic charts
    @FXML
    private VBox eval_box; // Horizontal container in scroll pane
    @FXML
    private ScrollPane scrollPane; // Scroll pane to display images
    @FXML
    private GridPane imgBox; // Horizontal container in scroll pane
    @FXML
    private VBox resultBox; // Container to hold results diagramm
    @FXML
    private BarChart resultChart;  // Histogram for results
    @FXML
    private Label loaded; // Label to show number of pictures loaded
    @FXML
    private Label f1; // Label to show f1
    @FXML
    private Label accuracy; // Label to show accuracy
    @FXML
    private Label recall; // Label to show recall
    @FXML
    private Label precision; // Label to show precision
    @FXML
    private Label falsePositiveRate; // Label to show false Positive Rate
    @FXML
    private Label falseNegativeRate; // Label to show false Negative Rate
    @FXML
    private Button get_results_btn; // Button to show results of evaluation
    @FXML
    private Button show_stats_btn; // Button to show statistics of evaluation

    // Data to process
    public ArrayList<String> addresses; // Adresses of images, which NN process during test/train
    public ArrayList<ImageView> imageViews; // Images showed in the window
    public List<File> files; // list of files loaded from  the File System
    public HashMap<String, Integer> predictions; // predictions of NN
    public HashMap<String, Integer> rightAnswers;  // right answers for the classification
    private double f1_score;
    private double precision_score;
    private double recall_score;
    private double accuracy_score;
    private double falsePositiveRate_score;
    private double falseNegativeRate_score;
    private int numProcessed;


    // Neural Network configs & params
    // VGG16 network object, allows to get it from every method in this class
    private ComputationGraph vgg16Transfer;

    // Normaize input data: mean (Expected value) = 0; variance = 1
    private NormalizerStandardize normalizer = new NormalizerStandardize();

    //Images are of format given by allowedExtension
    private final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    private int epochs = 3; // Number of epochs
    private int[] inputShape = new int[] {3, 224, 224}; // Size of input image: cohn-kanade - [1x640x490] -> convert to VGGFACE - [3, 224, 224]
    private int numLabels = 3; // Number of classes
    private int labelIndex = 1; // For images always = 1
    private long seed = 123; // Random seed
    private Random randNumGen = new Random(seed); // Randomizer
    private WorkspaceMode workspaceModeForTrain = WorkspaceMode.SEPARATE; // SEPARATE is slower, but used less memory. SINGLE is opposite.
    private WorkspaceMode workspaceModeForInference = WorkspaceMode.SINGLE; // SEPARATE is slower, but used less memory. SINGLE is opposite.
    private int batchSizeTraining = 5;
    private int batchSizeTesting = 5;
    private final String featureExtractionLayer = "fc7"; // Layer, under which all params are frozen


    // Variable, that hold the place in file system, where params of net are stored
    // If collectStats = true, then net is not trained yet and need to be trained
    // If collectStats = false, then params are already saved in file and needed to be loaded
    private static boolean collectStats = false;

    //TODO: Add new folder for user data!
    //    /ck
    private String trainDirAddr = new String("/ck");
    private String userDirAddr = new String("user_data");
    private String testDirAddr = new String("test_data");
    //Where to save the network. Note: the file is in .zip format - can be opened externally
    private File locationToSave = new File("EmotionsRecognitionNetwork.zip");
	private String trainFileName = new String("TrainLog.txt"); // File name to collect details about training process
	private String testFileName = new String("TestLog.txt"); // File name to collect details about testing process
	public  String statsFileName = new String("stats_1.csv"); // Name of file to collect stats while training
	// Create FileWriters
	FileWriter trainWriter;
	FileWriter testWriter;
	FileWriter statsWriter;
	// Titles of columns in csv stats file
	public String statsFirstLine = new String("Accuracy;Precision;Recall;F1;FPRate;FNRate"); 
	private final char CSV_SEPARATOR = ';';
	// File for logs
	private File trainFile; // File to collect details about training process
	private File testFile;  // File to collect details about testing process
	private File statsFile; // File for saving config and results 
    //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
    private boolean saveUpdater = true;
    // Iterators for the data
    private RecordReaderDataSetIterator trainIterator;
    private RecordReaderDataSetIterator testIterator;
    private RecordReaderDataSetIterator finalTestIterator; // Iterator for test on dataset data
    private RecordReaderDataSetIterator noiseTestIterator; // Iterator for test on data from the webcam with noise
    // Datasets data
    private DataSet trainingData;
    private DataSet testData;
    private DataSet finalTestData;
    private DataSet noiseTestData; //TODO: add noise test
    private DataSet userDataSet;  //TODO: add dealing with unlabeled data


    @Override
    public void init(){
        System.out.println("Application inits");
        try {
		// Create file for logging
		trainFile = new File(trainFileName);
		if(!trainFile.exists()) {
			System.out.println("\n\nTrain Log file does not exist. Create...");
			trainFile.createNewFile();
			System.out.println("\n\nTrain Log file created in folder: " + trainFile.getAbsolutePath());
		}
		else {
			System.out.println("\n\nTrain Log file created in folder: " + trainFile.getAbsolutePath());
		}
		testFile = new File(testFileName);
		if(!testFile.exists()) {
			System.out.println("\n\nTest Log file does not exist. Create...");
			testFile.createNewFile();
			System.out.println("\n\nTest Log file created in folder: " + testFile.getAbsolutePath());
		}
		else {
			System.out.println("\n\nTest Log file created in folder: " + testFile.getAbsolutePath());
		}
		// Create file for saving config and results 
		statsFile = new File(statsFileName);
		if(!statsFile.exists()) {
			System.out.println("\n\nStats file does not exist. Create...");
			statsFile.createNewFile();
			System.out.println("\n\nStats file created in folder: " + statsFile.getAbsolutePath());
		}
		else {
			System.out.println("\n\nStats file created in folder: " + statsFile.getAbsolutePath());
		}
		
		// Create FileWriters
		trainWriter = new FileWriter(trainFile, false);
		testWriter = new FileWriter(testFile, false);
		statsWriter = new FileWriter(statsFile, false);
		
		
        }  catch (Exception e) {
            System.err.println("Error in creating log files");
            e.printStackTrace();
        }
        
        try {
            super.init();
            trainWriter.write("---Initialization of Application.--- \nBuild model....");
            this.vgg16Transfer = configurate();
            trainWriter.write("Configuration created successfully!");
            System.out.println("Configuration created successfully!");
            vgg16Transfer.init();
            trainWriter.write("Neural Network initialized successfully!");
            trainWriter.write(vgg16Transfer.summary());
            System.out.println(vgg16Transfer.summary()); // Print changes config
        } catch (Exception e) {
            System.err.println("Error in configurating Neural Network");
            e.printStackTrace();
        }

        // Load data for training
        try {
        	trainWriter.write("\nStarted loading training data...");
            loadDataTraining(trainDirAddr);
        } catch (IOException e) {
            System.err.println("Error while loading training dataset");
            e.printStackTrace();
        }
        // Load data for testing
        try {
        	testWriter.write("Started loading training data...");
            loadDataTesting(testDirAddr);
        } catch (IOException e) {
            System.err.println("Error while loading testing dataset");
            e.printStackTrace();
        }

        /*
         * Check, if model is ready - then load it from file, then check it on test iterator.
         * Else - train it and save params to the specified file.
         */
        if(!collectStats) {
            try {
            	System.out.println("\n  Model is already pre-trained. Loading...");
            	testWriter.write("\n  Model is already pre-trained. Loading...");
            
                // Load model
                vgg16Transfer = load();
                testWriter.write("\n  Model loaded successfully!");
            } 
            catch (IOException e) {
                System.err.println("Error while writing to log file");
                e.printStackTrace();
            }
            catch (Exception e) {
                System.err.println("Error while loading Neural Network");
                e.printStackTrace();
            }
        }
        else  {
        	try {
				trainWriter.write("\n Model is not pre-trained.");
				System.out.println("\n  Model is not pre-trained.");
	            // Train NN
	            train();
				
			} 
            catch (IOException e) {
                System.err.println("Error while writing to log file");
                e.printStackTrace();
            }
        	catch (Exception e1) {
				System.err.println("Error while training Neural Network");
				e1.printStackTrace();
			}
           
            // Save NN to file
            try {
                System.out.println("\n Model is trained. Saving...");
                trainWriter.write("\n Model is trained. Saving...");
                save(vgg16Transfer);
                trainWriter.write("\n Model saved successfully!");
            } catch (Exception e) {
                System.err.println("Error while saving Neural Network");
                e.printStackTrace();
            }
        }
    }

    @Override
    public void start(Stage primaryStage) {
        AnchorPane root = new AnchorPane();
        Scene scene = new Scene(root,900,700);

        addresses = new ArrayList<>();

        scrollPane = new ScrollPane();
        // Setting a horizontal scroll bar is always display
        scrollPane.setHbarPolicy(ScrollPane.ScrollBarPolicy.ALWAYS);
        scrollPane.setPrefSize(874.0, 300.0);
        scrollPane.setPannable(false);
        AnchorPane.setTopAnchor(scrollPane, 50.0);
        AnchorPane.setLeftAnchor(scrollPane, 10.0);
        AnchorPane.setRightAnchor(scrollPane, 10.0);

        imgBox = new GridPane();
        imgBox.setHgap(10);
        imgBox.setPrefSize(872.0, 290.0);
        imgBox.setPadding(new Insets(5));
        scrollPane.setContent(imgBox);


        // Get images from files and display them
        ImageView imageView;
        int column = 0;
        imageViews = new ArrayList<>();
        for (File file : files) {
            try {
                imageView = new ImageView();
                imageView.setImage(fileToImage(file));
                imageView.setId(file.getAbsolutePath());
                // Setting the fit height and width of the image view
                imageView.setFitWidth(280);
                // Setting the preserve ratio of the image view
                imageView.setPreserveRatio(true);
                imageViews.add(imageView);
                imgBox.add(imageView, column, 0); // столбец= № изображения, строка=0
                column++;
            }
            catch (MalformedURLException e) {
                System.err.println("Error while loading image");
                e.printStackTrace();
            }
        }

        loaded = new Label("Loaded: ");
        loaded.setFont(new Font("Arial", 18));
        loaded.setPrefSize(500, 15);
//        loaded.setBorder(new Border(new BorderStroke(Color.GREEN, BorderStrokeStyle.SOLID, new CornerRadii(1), new BorderWidths(1))));
        AnchorPane.setTopAnchor(loaded, 15.0);
        AnchorPane.setLeftAnchor(loaded, 10.0);

        get_results_btn = new Button("Get Results");
        get_results_btn.setFont(new Font("Arial", 16.0));
        get_results_btn.setPrefSize(173.0, 33.0);
        get_results_btn.setOnAction(new EventHandler<ActionEvent>() {
        	
            @SuppressWarnings("static-access")
			@Override
            public void handle(ActionEvent event) {
                numProcessed = 0;
                loaded.setText("Loaded: " + addresses.size() + " pictures, processed: " + numProcessed + " pictures");
                Iterator<String> predictionsKeys = predictions.keySet().iterator();
                ObservableList<Node> images = imgBox.getChildren();
                int numImg = images.size();
                String addr = new String();
                Label prediction;
                HashMap<Integer, Label> predictionLabels = new HashMap<Integer, Label>();
                int columnIdx = 0;
                int predNum;
                while(predictionsKeys.hasNext())  {
                	addr = predictionsKeys.next();
                    for (Node im : images) {
                    	// addressesEquals(String addrFS - from ImageView, String addrNN - from metadata)
                    	if(addressesEquals(im.getId(), addr)) {
                    		columnIdx = imgBox.getColumnIndex(im);
                    		predNum = predictions.get(addr);
                    		switch (predNum) {
							case 0:
								prediction = new Label("Negative");
								prediction.setFont(new Font("Arial", 16.0));
								break;
							case 1:
								prediction = new Label("Neutral");
								prediction.setFont(new Font("Arial", 16.0));
								break;
							case 2:
								prediction = new Label("Positive");
								prediction.setFont(new Font("Arial", 16.0));
								break;

							default:
								prediction = new Label("Unrecognized!");
								prediction.setFont(new Font("Arial", 16.0));
								break;
							}
                    		// public void add(int index = column index, E element = label of prediction)
                            
                    		predictionLabels.put(columnIdx, prediction);
//                    		imgBox.add(prediction, columnIdx, 1);
                    	}
                    }
                }
                // Iterate over prediction labels and add then to imgBox
                for(int i = 0; i < numImg; i++)  {
                	imgBox.add(predictionLabels.get(i), i, 1);
                	imgBox.setHalignment(predictionLabels.get(i), HPos.CENTER);
                }
            }
        });

        show_stats_btn = new Button("Show Statistics");
        show_stats_btn.setFont(new Font("Arial", 16.0));
        show_stats_btn.setPrefSize(173.0, 33.0);
        show_stats_btn.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                try {
                    Label windowLabel = new Label("Evaluation results & statistics during training process");
                    windowLabel.setFont(new Font("Arial", 20.0));

                    Button show_nn_struct = new Button("View Neural Network Structure");
                    show_nn_struct.setFont(new Font("Arial", 16.0));
                    show_nn_struct.setPrefSize(300.0, 33.0);
                    show_nn_struct.setOnAction(new EventHandler<ActionEvent>() {
                        @Override
                        public void handle(ActionEvent event) {
                            ScrollPane scrollPaneNN = new ScrollPane();
                            ImageView nn_img = null;
                            try {
                                nn_img = new ImageView(fileToImage(new File(
                                        "J:\\Tests\\11.02.19 vers2\\EmotionsRecognitionVGG16_GUI\\src\\stats\\VGG16structure.png")));

//                                nn_img.setFitWidth(450);
                                // Setting the preserve ratio of the image view
                                nn_img.setPreserveRatio(true);
                                scrollPaneNN.setContent(nn_img);

                                Scene thirdScene = new Scene(scrollPaneNN, 435, 500);
                                // New window (Stage)
                                Stage newWindow2 = new Stage();
                                newWindow2.setTitle("Neural Network");
                                newWindow2.setScene(thirdScene);

                                // Set position of second window, related to primary window.
                                newWindow2.setX(primaryStage.getX() - 50);
                                newWindow2.setY(primaryStage.getY() - 50);

                                newWindow2.show();
                            }
                            catch (MalformedURLException e) {
                                e.printStackTrace();
                            }
                        }
                    });

                    Label f1Label = new Label("F1 score rating");
                    f1Label.setFont(new Font("Arial", 16.0));
                    ImageView f1_img = new ImageView(fileToImage(new File(
                            "J:\\Tests\\11.02.19 vers2\\EmotionsRecognitionVGG16_GUI\\src\\stats\\F1.JPG")));
                    f1_img.setFitWidth(450);
                    // Setting the preserve ratio of the image view
                    f1_img.setPreserveRatio(true);

                    Label precisionLabel = new Label("Precision score rating");
                    precisionLabel.setFont(new Font("Arial", 16.0));
                    ImageView precision_img = new ImageView(fileToImage(new File(
                            "J:\\Tests\\11.02.19 vers2\\EmotionsRecognitionVGG16_GUI\\src\\stats\\Precision.JPG")));
                    precision_img.setFitWidth(450);
                    // Setting the preserve ratio of the image view
                    precision_img.setPreserveRatio(true);

                    Label recallLabel = new Label("Recall score rating");
                    recallLabel.setFont(new Font("Arial", 16.0));
                    ImageView recall_img = new ImageView(fileToImage(new File(
                            "J:\\Tests\\11.02.19 vers2\\EmotionsRecognitionVGG16_GUI\\src\\stats\\Recall.JPG")));
                    recall_img.setFitWidth(450);
                    // Setting the preserve ratio of the image view
                    recall_img.setPreserveRatio(true);

                    Label rocLabel = new Label("ROC AUC score");
                    rocLabel.setFont(new Font("Arial", 16.0));
                    ImageView roc_img = new ImageView(fileToImage(new File(
                            "J:\\Tests\\11.02.19 vers2\\EmotionsRecognitionVGG16_GUI\\src\\stats\\ROC.JPG")));
                    roc_img.setFitWidth(450);
                    // Setting the preserve ratio of the image view
                    roc_img.setPreserveRatio(true);

                    GridPane gridpane = new GridPane();
                    gridpane.setPadding(new Insets(20));
                    gridpane.setHgap(25);
                    gridpane.setVgap(15);
                    gridpane.add(windowLabel, 0, 0); // столбец=0 строка=0
                    gridpane.add(show_nn_struct, 1, 0); // столбец=0 строка=0

                    gridpane.add(f1Label, 0, 1); // столбец=0 строка=1
                    gridpane.add(precisionLabel, 1, 1); // столбец=1 строка=1

                    gridpane.add(f1_img, 0, 2); // столбец=0 строка=2
                    gridpane.add(precision_img, 1, 2); // столбец=1 строка=2

                    gridpane.add(recallLabel, 0, 3); // столбец=0 строка=3
                    gridpane.add(rocLabel, 1, 3); // столбец=1 строка=3

                    gridpane.add(recall_img, 0, 4); // столбец=0 строка=4
                    gridpane.add(roc_img, 1, 4); // столбец=1 строка=4


                    Scene secondScene = new Scene(gridpane, 1000, 750);
                    // New window (Stage)
                    Stage newWindow = new Stage();
                    newWindow.setTitle("Statistics");
                    newWindow.setScene(secondScene);

                    // Set position of second window, related to primary window.
                    newWindow.setX(primaryStage.getX() - 50);
                    newWindow.setY(primaryStage.getY() - 50);

                    newWindow.show();
                }
                catch (MalformedURLException e) {
                    e.printStackTrace();
                }
            }
        });

        f1 = new Label("F1: ");
        accuracy = new Label("Accuracy: ");
        recall = new Label("Recall: ");
        precision = new Label("Precision: ");
        falseNegativeRate = new Label("False Negative Rate: ");
        falsePositiveRate = new Label("False Positive Rate: ");
        f1.setFont(new Font("Arial", 16.0));
        accuracy.setFont(new Font("Arial", 16.0));
        recall.setFont(new Font("Arial", 16.0));
        precision.setFont(new Font("Arial", 16.0));
        falseNegativeRate.setFont(new Font("Arial", 16.0));
        falsePositiveRate.setFont(new Font("Arial", 16.0));

        eval_box = new VBox();
        eval_box.setAlignment(Pos.TOP_CENTER);
        eval_box.setSpacing(15);
        eval_box.setPrefSize(300.0, 300.0);
        eval_box.setPadding(new Insets(5));
        eval_box.getChildren().addAll(show_stats_btn, get_results_btn, f1, accuracy,
                recall, precision, falseNegativeRate, falsePositiveRate) ;
        AnchorPane.setBottomAnchor(eval_box, 40.0);
        AnchorPane.setLeftAnchor(eval_box, 30.0);

        resultBox = new VBox();
        resultBox.setPrefSize(530.0, 300.0);
        AnchorPane.setBottomAnchor(resultBox, 40.0);
        AnchorPane.setRightAnchor(resultBox, 10.0);
        resultBox.setAlignment(Pos.TOP_CENTER);
//        resultBox.setBorder(new Border(new BorderStroke(Color.GREEN, BorderStrokeStyle.SOLID, new CornerRadii(1), new BorderWidths(1))));

        // Create a BarChart
        CategoryAxis xAxis = new CategoryAxis();
        xAxis.setLabel("Emotions");
        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Number of images");
        resultChart = new BarChart<String, Number>(xAxis, yAxis);
        resultChart.setTitle("Statistics of predictions vs. right answers");
        resultChart.setPrefSize(530.0, 300.0);
        resultBox.getChildren().add(resultChart);

        root.getChildren().addAll(loaded, scrollPane, eval_box, resultBox);


        primaryStage.setScene(scene);
        primaryStage.show();

        //Allow JavaFX do to it's thing, Initialize the Neural network when it feels like it.
        Platform.runLater(this::test);

    }


    /**
     * Convert file to image and returns it
     * @param file
     * @return
     * @throws MalformedURLException
     */
    private static Image fileToImage(File file) throws MalformedURLException {

        String localUrl = file.toURI().toURL().toString(); // Get URL of file
        Image image = new Image(localUrl);  // Create new image
        return image;
    }

    // Load files with images from file system
    public void loadFiles(String location) {
        files = new ArrayList<>();
        try {
            // определяем объект для каталога
            File dir = new File(location);
            String[] listOfFiles = dir.list();
            if(dir.isDirectory()) {
            	System.out.println("Dir: " + dir.getAbsolutePath());
            	testWriter.write("Dir: " + dir.getAbsolutePath());
            }
            for(File i : dir.listFiles()) {
                System.out.println("SubDir: " + i.getAbsolutePath());
                for(File j : i.listFiles()) {
                    files.add(new File(j.getAbsolutePath()));
                    System.out.println("File: " + j.getAbsolutePath() + "  is loaded");
                    testWriter.write("File: " + j.getAbsolutePath() + "  is loaded");
                }
            }
        }
        catch(NullPointerException e)  {
            System.err.println("Error while loading file");
            e.printStackTrace();
        } catch (IOException e) {
        	System.err.println("Error while writing to log file");
			e.printStackTrace();
		}
    }

    /**
     * Method, in which:
     * - Create required configuration, print, log and initialize neural net
     * @return
     * @throws IOException
     */
    private ComputationGraph configurate() throws IOException  {
        // Load Zoo model VGG16
        ZooModel zooModel = new VGG16();
        trainWriter.write("\nLoading org.deeplearning4j.transferlearning.vgg16...\n");
        System.out.println("\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.VGGFACE);
        trainWriter.write("\nLoaded model overview: " + vgg16.summary());
        
        // Create Fine tune configuration, which will modify params of all layers, that are not frozen
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Nesterovs(5e-5))
                .seed(123)
                .build();

        /*
         * Change to required config and print it.
         * Therefore the last layer (as seen when printing the summary) is a dense layer and not an output layer with a loss function.
         * Therefore to modify nOut of an output layer we delete the layer vertex,
         * keeping it’s connections and add back in a new output layer with the same name,
         * a different nOut, the suitable loss function etc etc.
         * */
        vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
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
        trainWriter.write("\nCorrected model overview: " + vgg16Transfer.summary());
        System.out.println("\nCorrected model overview: " + vgg16Transfer.summary()); // Print changes config

        return vgg16Transfer;
    }

    /*
     * Method that carries training and testing process if NN is not pre-trained
     */
    private void train() throws IOException  {

        // Set Listeners, to see results of evaluation during training process and testing
        vgg16Transfer.setListeners(new ScoreIterationListener(100));
        trainWriter.write("\n  Train model....");
        System.out.println("\n  Train model....");
        int iterationsCounter = 0;
        // Go through all data "epochs" - number of times
        for(int n = 0; n < epochs; n++)  {
            System.out.println(String.format("Epoch %d started training", n + 1));
            trainWriter.write(String.format("Epoch %d started training", n + 1));

            //Reset training iterator to the new epoch
            trainIterator.reset();
            // Go through all data once, till it's end
            while (trainIterator.hasNext()) {
                iterationsCounter++;
                trainingData = trainIterator.next();
                normalizer.fit(trainingData);
                vgg16Transfer.fit(trainingData);
                System.out.println("*** Completed iteration number # " + iterationsCounter + "  ***");
                trainWriter.write("*** Completed iteration number # " + iterationsCounter + "  ***");

            }
            System.out.println(String.format("Epoch %d finished training", n + 1));
            trainWriter.write("*** Completed iteration number # " + iterationsCounter + "  ***");
            // Get results and check them
            Evaluation eval = new Evaluation(numLabels);
            testIterator.reset();
            while(testIterator.hasNext()) {
                testData = testIterator.next();
                normalizer.fit(testData);
                INDArray features = testData.getFeatures();
                INDArray labels = testData.getLabels();
                INDArray predicted = vgg16Transfer.outputSingle(features);
                eval.eval(labels, predicted);
            }
            System.out.println(String.format("Evaluation on test data - [Epoch %d] [Accuracy: %.3f, P: %.3f, R: %.3f, F1: %.3f] ",
                    n + 1, eval.accuracy(), eval.precision(), eval.recall(), eval.f1()));
            System.out.println(eval.stats());
            System.out.println(eval.confusionToString());
            trainWriter.write(String.format("Evaluation on test data - [Epoch %d] [Accuracy: %.3f, P: %.3f, R: %.3f, F1: %.3f] ",
                    n + 1, eval.accuracy(), eval.precision(), eval.recall(), eval.f1()));
            trainWriter.write(eval.stats());
            trainWriter.write(eval.confusionToString());
        }
        System.out.println("\n  *** Training finished! *** ");
        trainWriter.write("\n  *** Training finished! *** ");
    }
    /*
     * Method that carries only testing if NN is already pre-trained
     * and loaded from file or when training ended
     */
    private void test()  {
    	try {
			testWriter.write("Testing started...");
		
			predictions = new HashMap<>();
        	rightAnswers = new HashMap<>();


        	// Get results and check them
        	Evaluation eval = new Evaluation(numLabels);
        	finalTestIterator.reset();

        	int metaDataCounter = 0;
        	int addrCounter = 0;

        	while(finalTestIterator.hasNext()) {
            	// If iterator has next dataset
            	finalTestData = finalTestIterator.next();
            	// Get meta-data from this dataset

            	@SuppressWarnings("rawtypes")
            	List<?> exampleMetaData = finalTestData.getExampleMetaData();
            	Iterator<?> exampleMetaDataIterator = exampleMetaData.iterator();
            	testWriter.write("\n  Metadata from dataset #" + metaDataCounter + ":\n");
            	System.out.println("\n  Metadata from dataset #" + metaDataCounter + ":\n");

            	// Normalize data
            	normalizer.fit(finalTestData);

            	// Count processed images
            	numProcessed = (metaDataCounter + 1) * batchSizeTesting;
            	loaded.setText("Loaded and processed: " + numProcessed + " pictures");

            	INDArray features = finalTestData.getFeatures();
            	INDArray labels = finalTestData.getLabels();
            	System.out.println("\n  Right labels #" + metaDataCounter + ":\n");
            	testWriter.write("\n  Right labels #" + metaDataCounter + ":\n");
            	// Get right answers of NN for every input object
            	int[][] rightLabels = labels.toIntMatrix();
            	for (int i = 0; i < rightLabels.length; i++) {
                	RecordMetaDataURI metaDataUri = (RecordMetaDataURI) exampleMetaDataIterator.next();
                	// Print address of image
                	System.out.println(metaDataUri.getLocation());
                	for (int j = 0; j < rightLabels[i].length; j++) {
                    	if(rightLabels[i][j] == 1) {
                        	//public V put(K key, V value) -> key=address, value=right class label
                        	rightAnswers.put(metaDataUri.getLocation(), j);
                        	this.addresses.add(metaDataUri.getLocation());
                    	}
                	}
            	}
            	System.out.println("\nRight answers:");
            	testWriter.write("\nRight answers:");
            	// Print right answers
            	for(Map.Entry<String, Integer> answer : predictions.entrySet()){
            		testWriter.write(String.format("Address: %s  Right answer: %s \n", answer.getKey(), answer.getValue().toString()));
                	System.out.printf(String.format("Address: %s  Right answer: %s \n", answer.getKey(), answer.getValue().toString()));
            	}

            	// Evaluate on the test data
            	INDArray predicted = vgg16Transfer.outputSingle(features);
            	int predFoundCounter = 0;
            	System.out.println("\n Labels predicted #" + metaDataCounter + ":\n");
            	testWriter.write("\n Labels predicted #" + metaDataCounter + ":\n");
            	// Get predictions of NN for every input object
            	int[][] labelsPredicted = predicted.toIntMatrix();
            	for (int i = 0; i < labelsPredicted.length; i++) {
                	for (int j = 0; j < labelsPredicted[i].length; j++) {
                    	predFoundCounter++;
                    	if(labelsPredicted[i][j] == 1) {
                        	//public V put(K key, V value) -> key=address, value=predicted class label
                        	predFoundCounter = 0;
                        	this.predictions.put(this.addresses.get(addrCounter), j);
                    	}
                    	else {
                        	if (predFoundCounter == 3)  {
                            	// To fix bug when searching positive predictions
                            	this.predictions.put(this.addresses.get(addrCounter), 2);
                        	}
                    	}
                	}
                	addrCounter++;
            	}
            	System.out.println("\nPredicted:");
            	testWriter.write("\nPredicted:");
            	// Print predictions
            	for(Map.Entry<String, Integer> pred : rightAnswers.entrySet()){
                	System.out.printf("Address: %s Predicted answer: %s \n", pred.getKey(), pred.getValue().toString());
                	testWriter.write(String.format("Address: %s Predicted answer: %s \n", pred.getKey(), pred.getValue().toString()));
            	}
            	metaDataCounter++;

            	eval.eval(labels, predicted);
        	}

        	System.out.println("\n\n Cheack loaded model on test data...");
        	System.out.println(String.format("Evaluation on test data - [Accuracy: %.3f, P: %.3f, R: %.3f, F1: %.3f] ",
        			eval.accuracy(), eval.precision(), eval.recall(), eval.f1()));
        	System.out.println(eval.stats());
        	System.out.println(eval.confusionToString());
        	testWriter.write("\n\n Cheack loaded model on test data...");
        	testWriter.write(String.format("Evaluation on test data - [Accuracy: %.3f, P: %.3f, R: %.3f, F1: %.3f] ",
        			eval.accuracy(), eval.precision(), eval.recall(), eval.f1()));
        	testWriter.write(eval.stats());
        	testWriter.write(eval.confusionToString());

        	// Save test rates
        	this.f1_score = eval.f1();
        	this.recall_score = eval.recall();
        	this.accuracy_score = eval.accuracy();
        	this.precision_score = eval.precision();
        	this.falseNegativeRate_score = eval.falseNegativeRate();
        	this.falsePositiveRate_score = eval.falsePositiveRate();

        	this.f1.setText("F1 = " + String.format("%.4f", f1_score));
        	this.recall.setText("Recall = " + String.format("%.4f", recall_score));
        	this.accuracy.setText("Accuracy = " + String.format("%.4f", accuracy_score));
        	this.precision.setText("Precision = " + String.format("%.4f", precision_score));
        	this.falseNegativeRate.setText("False Negative Rate = " + String.format("%.4f", falseNegativeRate_score));
        	this.falsePositiveRate.setText("False Positive Rate = " + String.format("%.4f", falsePositiveRate_score));
        
        	testWriter.write("F1 = " + String.format("%.4f", f1_score));
        	testWriter.write("Recall = " + String.format("%.4f", recall_score));
        	testWriter.write("Accuracy = " + String.format("%.4f", accuracy_score));
        	testWriter.write("Precision = " + String.format("%.4f", precision_score));
        	testWriter.write("False Negative Rate = " + String.format("%.4f", falseNegativeRate_score));
        	testWriter.write("False Positive Rate = " + String.format("%.4f", falsePositiveRate_score));

        	showBarChart();
    	} catch (IOException e) {
    		System.err.println("Error while writing to log file");
			e.printStackTrace();
		}
    }

    /**
     * Load user chosen data
     * @throws IOException
     */
    private void loadUserData(String location) throws IOException {
        /**
         * Create dataset: public DataSet(INDArray data, INDArray labels)
         * Create arrays:
         * data(1 - single picture to single iteration, number of pixels 3*224*224)
         * labels(single label to single picture)
         */

        INDArray data = Nd4j.zeros(1, inputShape[1]*inputShape[2]*inputShape[0]);
        INDArray labels = Nd4j.zeros(1, 1);
    }



    /**
     * Load data for testing
     * @throws IOException
     */
    private void loadDataTesting(String location) throws IOException {
    	
        File parentDir = new ClassPathResource(location).getFile();
        testWriter.write("Data folder found.");

        // Divide all data to training and testing datasets
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
        testWriter.write("Data split created.");

        // Create ParentPathLabelGenerator, that will parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        testWriter.write("Label generators created.");

        ImageRecordReader testRecordReader = new ImageRecordReader(inputShape[1],inputShape[2],inputShape[0], labelMaker);
        testRecordReader.initialize(filesInDir);
        testWriter.write("Data iterators created successfully.");

        // Get the number of labels, ImageRecordReader founded in dir and check it
        int outputNum = testRecordReader.numLabels();
        finalTestIterator = new RecordReaderDataSetIterator(testRecordReader, batchSizeTesting, labelIndex, outputNum);
        finalTestIterator.setCollectMetaData(true);
        testWriter.write("Data iterators setted to collect metadata.");

        // Load files with images from file system
        testWriter.write("Started loading testing files...");
        loadFiles("J:\\Tests\\11.02.19 vers2\\EmotionsRecognitionVGG16_GUI\\src\\test_data");
        testWriter.write("Testing files loaded successfully!");
    }

    /**
     * Load data for trainig & testing
     * @throws IOException
     */
    private void loadDataTraining(String location) throws IOException {

        File parentDir = new ClassPathResource(location).getFile();
    	trainWriter.write("Data folder found.");

        // Divide all data to training and testing datasets
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
        
        // Create ParentPathLabelGenerator, that will parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        trainWriter.write("Label generators created.");
        
        // Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainDataInputSplit = filesInDirSplit[0];
        InputSplit testDataInputSplit = filesInDirSplit[1];
        trainWriter.write("Data splited to train and test parts as 80% and 20%.");
        
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
        trainWriter.write("Data iterators created successfully.");
        
        // Get the number of labels, ImageRecordReader founded in dir and check it
        int outputNum = trainRecordReader.numLabels();

        // Init iterators with datasets
        trainIterator = new RecordReaderDataSetIterator(trainRecordReader, batchSizeTraining, labelIndex, outputNum);
        trainIterator.setCollectMetaData(true);
        testIterator = new RecordReaderDataSetIterator(testRecordReader, batchSizeTesting, labelIndex, outputNum);
        testIterator.setCollectMetaData(true);
        trainWriter.write("Data iterators setted to collect metadata.");
    }

    /**
     * Method, that get data from CNN and draw bar chart of results vs. right amswers
     */
    public void showBarChart()  {

        // Count classes
        int neutralCountPred = 0;
        int negativeCountPred = 0;
        int positiveCountPred = 0;

        int neutralCountAnsw = 0;
        int negativeCountAnsw = 0;
        int positiveCountAnsw = 0;

        for(Integer pred : predictions.values())  {
            switch (pred.intValue()) {
                case 0:
                    negativeCountPred++;
                    break;
                case 1:
                    neutralCountPred++;
                    break;
                case 2:
                    positiveCountPred++;
                    break;
                default:
                    System.err.println("Illegal class index");
                    break;
            }
        }
        System.out.printf("PREDICTED \nnegativeCountPred = %d, neutralCountPred = %d, positiveCountPred = %d", negativeCountPred,
                neutralCountPred, positiveCountPred);

        for(Integer answer : rightAnswers.values())  {
            switch (answer.intValue()) {
                case 0:
                    negativeCountAnsw++;
                    break;
                case 1:
                    neutralCountAnsw++;
                    break;
                case 2:
                    positiveCountAnsw++;
                    break;
                default:
                    System.err.println("Illegal class index");
                    break;
            }
        }
        System.out.printf("\nRIGHT ANSWERS \nnegativeCountAnsw = %d, neutralCountAnsw = %d, positiveCountAnsw = %d", negativeCountAnsw,
               neutralCountAnsw, positiveCountAnsw);

        // Predicted classes
        XYChart.Series<String, Number> dataSeries1 = new XYChart.Series();
        dataSeries1.setName("Predicted");
        dataSeries1.getData().add(new XYChart.Data("Neutral", neutralCountPred));
        dataSeries1.getData().add(new XYChart.Data("Positive", positiveCountPred));
        dataSeries1.getData().add(new XYChart.Data("Negative", negativeCountPred));
        resultChart.getData().add(dataSeries1);

        // Predicted classes
        XYChart.Series<String, Number> dataSeries2 = new XYChart.Series();
        dataSeries2.setName("Right answers");
        dataSeries2.getData().add(new XYChart.Data("Neutral", neutralCountAnsw));
        dataSeries2.getData().add(new XYChart.Data("Positive", positiveCountAnsw));
        dataSeries2.getData().add(new XYChart.Data("Negative", negativeCountAnsw));
        resultChart.getData().add(dataSeries2);

    }

    // Save model to file
    private void save(ComputationGraph net) throws Exception {
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);
    }

    // Load model from file
    private ComputationGraph load() throws Exception {
        ComputationGraph restored = ModelSerializer.restoreComputationGraph(locationToSave);
        return restored;
    }
    
    // Parse and compare addresses from file system and NN metadata
    private boolean addressesEquals(String addrFS, String addrNN)  {
		String[] strs = addrNN.split("/");
		String addrFS1 = new String(addrFS);
		if(addrFS1.endsWith(strs[strs.length - 1]))  {
			return true;
		}
		else {
			return false;
		}
    }
        

    public static void main(String[] args) {
        launch(args);
    }
}


