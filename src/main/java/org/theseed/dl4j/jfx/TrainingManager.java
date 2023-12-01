/**
 *
 */
package org.theseed.dl4j.jfx;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.commons.lang3.StringUtils;
import org.theseed.basic.ICommand;
import org.theseed.dl4j.train.CrossValidateProcessor;
import org.theseed.dl4j.train.ITrainReporter;
import org.theseed.dl4j.train.ITrainingProcessor;
import org.theseed.dl4j.train.ModelType;
import org.theseed.dl4j.train.SearchProcessor;
import org.theseed.io.LineReader;
import org.theseed.jfx.BackgroundTask;
import org.theseed.jfx.BaseController;
import org.theseed.jfx.IBackgroundController;
import org.theseed.jfx.ResizableController;
import org.theseed.join.JoinDialog;
import org.theseed.utils.Parms;

import javafx.application.Platform;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.geometry.Insets;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundFill;
import javafx.scene.layout.CornerRadii;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;

/**
 * This is the operating class for the main DL4J GUI window.  From here you can do load a model, do searches,
 * cross-validate, edit the parms, and view the log and the validation data.
 *
 * @author Bruce Parrello
 *
 */
public class TrainingManager extends ResizableController implements ITrainReporter, IBackgroundController  {

    // FIELDS
    /** current model directory */
    private File modelDirectory;
    /** TRUE if a model file exists */
    private File modelFile;
    /** current model type */
    private ModelType modelType;
    /** current model's parameter file */
    private File parmFile;
    /** number of epochs displayed (owned by platform thread) */
    private int numDisplayed;
    /** number of epochs queued (owned by background thread) */
    private int numSubmitted;
    /** label column index for training file analysis */
    private int labelIdx;
    /** ID column name for training file view */
    private String idColName;
    /** list of label names */
    private List<String> labelNames;
    /** current background task */
    private BackgroundTask backgrounder;
    /** current progress bar background color */
    private Color barColor;
    /** stop flag:  set to TRUE to abort the background task */
    private boolean stopFlag;
    /** pattern for finding line terminators in log messages */
    public static final Pattern LINE_END = Pattern.compile("\\r?\\n");
    /** system line separator */
    public static final String LINE_SEP = System.getProperty("line.separator");
    /** upper limit for progress bar */
    private static final double LOG10_UPPER = 1;
    /** lower limit for progress bar */
    private static final double LOG10_LOWER = -10;
    /** minimum displayable score */
    private static final double MIN_SCORE = Math.pow(10.0, LOG10_LOWER);
    /** color for a score over 1.0 */
    private static final Color BAD_COLOR = Color.RED;
    /** color for a score from 0.1 to 1.0 */
    private static final Color WARN_COLOR = Color.ORANGE;
    /** normal color */
    private static final Color NORMAL_COLOR = Color.TURQUOISE;
    /** good color */
    private static final Color GOOD_COLOR = Color.LIGHTGREEN;

    // CONTROLS

    /** model directory display */
    @FXML
    private TextField txtModelDirectory;

    /** get-directory button */
    @FXML
    private Button btnGetDirectory;

    /** model type description */
    @FXML
    private Label lblModelType;

    /** text display for current epoch */
    @FXML
    private TextField txtEpoch;

    /** text display for current score */
    @FXML
    private TextField txtScore;

    /** text display for current rating */
    @FXML
    private TextField txtRating;

    /** text display for best epoch */
    @FXML
    private TextField txtBestEpoch;

    /** button to start training search */
    @FXML
    private Button btnTrainingSearch;

    /** button to start cross-validation */
    @FXML
    private Button btnCrossValidate;

    /** button to view log */
    @FXML
    private Button btnViewLog;

    /** button to view results */
    @FXML
    private Button btnViewResults;

    /** button to make predictions */
    @FXML
    private Button btnPredict;

    /** abort button */
    @FXML
    private Button btnAbortCommand;

    /** message buffer */
    @FXML
    private TextField txtMessageBuffer;

    /** results pane */
    @FXML
    private TextArea txtResultsPane;

    /** simulated progress bar */
    @FXML
    private VBox barProgress;

    /** simulated progress bar container */
    @FXML
    private VBox barContainer;

    /** button to convert input */
    @FXML
    private Button btnConvert;

    /** button to view training file */
    @FXML
    private Button btnTrainingView;

    /** number of folds for cross-validation */
    @FXML
    private ChoiceBox<String> kFoldChooser;

    /**
     * Initialize the local fields.
     */
    public TrainingManager() {
        super(200, 200, 1000, 600);
        this.modelDirectory = null;
    }

    @Override
    public String getIconName() {
        return "fig-gear-16.png";
    }

    @Override
    public String getWindowTitle() {
        return "DL4J Training Manager";
    }


    public void init() {
        // Denote there is no background task running.
        this.backgrounder = null;
        this.stopFlag = false;
        // Get the last-saved model directory.
        String dirName = this.getPref("modelDirectory", "");
        File newDir = null;
        if (! dirName.isEmpty())
            newDir = new File(dirName);
        else
            newDir = new File(System.getProperty("user.dir"));
        // Only proceed if the directory has a parms.prm file.  We don't want to pop up the meta-dialog
        // here.  If there is no parms.prm, we assume the directory has been erased.
        File parmsFile = new File(newDir, "parms.prm");
        if (! parmsFile.exists()) {
            this.setState(false);
        } else {
            try {
                boolean ok = this.analyzeModelDirectory(newDir);
                if (! ok)
                    this.modelDirectory = null;
            } catch (IOException e) {
                this.setState(false);
                BaseController.messageBox(Alert.AlertType.ERROR, "Error Reading Model Directory", e.toString());
            }
        }
        // Set the cross-validation fold number.
        this.kFoldChooser.getItems().addAll("5", "9", "10", "13");
        String kFoldDefault = this.getPref("kFold", "10");
        this.kFoldChooser.setValue(kFoldDefault);
    }

    /**
     * Process a request for a new model directory.  If the directory is updated,
     * we will set up for the new model.
     *
     * @param event		event descriptor
     */
    @FXML
    private void selectModelDirectory(Event event) {
        DirectoryChooser chooser = new DirectoryChooser();
        chooser.setTitle("Select a Model Directory");
        // Start at the old model directory, or the current directory if there was none.
        File curDir = this.modelDirectory;
        if (curDir == null) curDir = new File(System.getProperty("user.dir"));
        chooser.setInitialDirectory(curDir);
        boolean found = false;
        while (! found) {
            curDir = chooser.showDialog(this.getStage());
            if (curDir == null) {
                // Here the user is canceling out.
                found = true;
            } else {
                // Here we have a possible directory.
                try {
                    // Make sure that if we retry, we start from here.
                    chooser.setInitialDirectory(curDir);
                    // Test the directory.
                    found = this.analyzeModelDirectory(curDir);
                    if (! found) {
                        // Directory was invalid.  Try again.
                        BaseController.messageBox(Alert.AlertType.WARNING, "Error Changing Model Directory",
                                curDir + " does not have a training.tbl and a labels.txt.");
                    }
                } catch (IOException e) {
                    BaseController.messageBox(Alert.AlertType.ERROR, "Error Changing Model Directory", e.toString());
                }
            }
        }
    }

    /**
     * Stop the current background task.
     *
     * @param event		event descriptor
     */
    @FXML
    private void abortCommand(Event event) {
        if (this.backgrounder != null) {
            this.stopFlag = true;
        }

    }

    /**
     * Button event to bring up the parameter dialog and run a training search if it is successful.
     *
     * @param event		event descriptor
     */
    @FXML
    private void runSearch(Event event) {
        try {
            if (! this.isValidTraining()) {
                BaseController.messageBox(Alert.AlertType.ERROR, "Uninitialized Training",
                        "Training file has no data.  You may need a data conversion first.");
            } else {
                Stage dialogStage = new Stage();
                ParmDialog dialog = (ParmDialog) BaseController.loadFXML(App.class, "ParmDialog", dialogStage);
                dialog.init(this.parmFile, this.modelType);
                dialogStage.showAndWait();
                if (dialog.getResult()) {
                    ICommand processor = new SearchProcessor(this);
                    String[] parms = new String[] { "-t", this.modelType.toString(), this.modelDirectory.toString() };
                    executeCommand(processor, "SEARCH", parms);
                }
            }
        } catch (IOException e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Loading Parameter Dialog", e.toString());
        }
    }

    /**
     * @return TRUE if the training file has data
     */
    private boolean isValidTraining() {
        boolean retVal = false;
        File trainFile = new File(this.modelDirectory, "training.tbl");
        try (LineReader trainStream = new LineReader(trainFile)) {
            // Skip the header.
            trainStream.next();
            // If there is at least one more record, we are ok.
            if (trainStream.hasNext())
                retVal = true;
        } catch (IOException e) {
            // Ignore the error.  An IO error means the training file is not ready.
        }
        return retVal;
    }

    /**
     * Button event to display the log view.
     *
     * @param event		event descriptor
     */
    @FXML
    private void viewLog(Event event) {
        try {
            Stage logStage = new Stage();
            LogViewer logViewer = (LogViewer) BaseController.loadFXML(App.class, "LogViewer", logStage);
            File logFile = new File(this.modelDirectory, "trials.log");
            logViewer.init(logFile, this.txtModelDirectory.getText());
            logStage.show();
        } catch (IOException e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Loading Log Viewer", e.toString());
        }
    }

    /**
     * Button event to display the results.
     *
     * @param event		event descriptor
     */
    @FXML
    private void showModelResults(Event event) {
        try {
            Stage resultStage = new Stage();
            // Get the training processor and set it up for predictions.
            ITrainingProcessor processor = ModelType.create(this.modelType);
            processor.setProgressMonitor(this);
            boolean ok = processor.initializeForPredictions(this.modelDirectory);
            if (ok) {
                // Compute the type of result display.
                String displayType = this.modelType.getDisplayType();
                // Create and display the result dialog.
                ResultDisplay resultDialog = (ResultDisplay) BaseController.loadFXML(App.class, "ResultDisplay", resultStage);
                resultDialog.init(processor, displayType);
                resultStage.showAndWait();
            } else {
                BaseController.messageBox(Alert.AlertType.ERROR, "Invalid Parameter File", "Parameter file not set up for evaluation.");
            }
        } catch (IOException e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Loading Result Display", e.toString());
        }
    }

    /**
     * Button event to make predictions.
     *
     * @param event		event descriptor
     */
    @FXML
    private void showPredict(Event event) {
        try {
            Stage predictStage = new Stage();
            PredictDialog predictDialog = (PredictDialog) BaseController.loadFXML(App.class, "PredictDialog", predictStage);
            predictDialog.init(this.modelDirectory, this.modelType);
            predictStage.showAndWait();
            // Check for a join request.
            File joinFile = predictDialog.isJoinRequested();
            if (joinFile != null)
                this.invokeJoinDialog(joinFile);
        } catch (Exception e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Making Predictions", e.toString());
        }
    }

    /**
     * Button event to join files.
     *
     * @param event		event descriptor
     */
    @FXML
    private void showJoinDialog(Event event) {
        try {
            invokeJoinDialog(null);
        } catch (Exception e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Joining Files", e.toString());
        }
    }

    /**
     * Invoke the join dialog.
     *
     * @param initFile	initial input file, or NULL if none
     *
     * @throws IOException
     */
    private void invokeJoinDialog(File initFile) throws IOException {
        Stage joinStage = new Stage();
        JoinDialog joinDialog = (JoinDialog) BaseController.loadFXML(JoinDialog.class, "JoinDialog", joinStage);
        joinDialog.init(this.modelDirectory, initFile);
        joinStage.showAndWait();
    }

    /**
     * Button event to invoke cross-validation.
     *
     * @param event		event descriptor
     */
    @FXML
    private void runXValidate(Event event) {
        ICommand processor = new CrossValidateProcessor(this);
        try {
            // Save the k-fold value.
            this.setPref("kFold", this.kFoldChooser.getValue());
            String idCol = this.getIdCol();
            List<String> parmList = new ArrayList<String>(10);
            parmList.add("-t");
            parmList.add(modelType.toString());
            if (idCol != null) {
                parmList.add("--id");
                parmList.add(idCol);
            }
            parmList.add("-k");
            parmList.add(this.kFoldChooser.getValue());
            parmList.add(this.modelDirectory.toString());
            String[] parms = new String[parmList.size()];
            parms = parmList.toArray(parms);
            this.executeCommand(processor, "Cross-Validate", parms);
        } catch (IOException e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Reading Parm File", e.toString());
        }
    }

    /**
     * @return the ID column for the current model, or NULL if there is none
     *
     * @throws IOException
     */
    private String getIdCol() throws IOException {
        String retVal = null;
        Parms parms = new Parms(parmFile);
        String[] cols = StringUtils.split(parms.getValue("--metaCols"), ',');
        if (cols.length >= 1)
            retVal = cols[0];
        return retVal;
    }

    /**
     * Process a change to a new model directory.  We update the controls and determine the model type.
     *
     * @param newDir	proposed new model directory
     *
     * @return TRUE if the directory is valid, else FALSE
     *
     * @throws IOException
     */
    protected boolean analyzeModelDirectory(File newDir) throws IOException {
        boolean retVal = false;
        if (newDir == null) {
            // Here we have no model directory.
            this.modelDirectory = null;
            this.setState(false);
            retVal = true;
        } else {
            // Now we need to determine what type of model this is.
            File labelFile = new File(newDir, "labels.txt");
            File trainFile = new File(newDir, "training.tbl");
            File dataFile = new File(newDir, "data.tbl");
            if (dataFile.exists() && ! trainFile.exists())
                this.createTrainingFile(trainFile, dataFile);
            if (labelFile.exists() && trainFile.exists()) {
                this.modelDirectory = newDir;
                this.modelFile = new File(newDir, "model.ser");
                txtModelDirectory.setText(newDir.getName());
                this.setPref("modelDirectory", newDir.getAbsolutePath());
                this.labelNames = LineReader.readList(labelFile);
                try (LineReader reader = new LineReader(trainFile)) {
                    String header = reader.next();
                    // If all of the labels are in this header line, it is a regression model.
                    List<String> headers = Arrays.asList(StringUtils.split(header, '\t'));
                    int count = 0;
                    for (String head : headers) {
                        if (this.labelNames.contains(head)) count++;
                    }
                    if (count == this.labelNames.size()) {
                        // Here we have a regression model.
                        modelType = ModelType.REGRESSION;
                    } else {
                        // Check for a marker file.
                        File marker = new File(newDir, "decider.txt");
                        if (marker.exists())
                            modelType = ModelType.DECISION;
                        else
                            modelType = ModelType.CLASS;
                    }
                    configureType();
                    // Create a processor.
                    ITrainingProcessor processor = ModelType.create(modelType);
                    // Set the defaults.
                    processor.setAllDefaults();
                    // Check for a parms.prm file.
                    parmFile = new File(this.modelDirectory, "parms.prm");
                    if (! parmFile.exists()) {
                        // Count the training set.
                        int size = 0;
                        while (reader.hasNext()) {
                            reader.next();
                            size++;
                        }
                        // Pull up the meta-column dialog.
                        List<String> availableHeaders = processor.computeAvailableHeaders(headers, this.labelNames);
                        String[] metaCols = this.findMetaColumns(availableHeaders);
                        processor.setMetaCols(metaCols);
                        if (metaCols.length > modelType.metaLabel())
                            processor.setIdCol(metaCols[0]);
                        // Compute the size-related parms.
                        int inputs = availableHeaders.size() - metaCols.length;
                        processor.setSizeParms(size, inputs);
                        // Write the parm file.
                        processor.writeParms(parmFile);
                    }
                    // Compute the label column index.
                    this.labelIdx = processor.getLabelIndex(this.modelDirectory);
                    // If we made it here without any errors, we can enable the buttons.
                    this.setState(true);
                    retVal = true;
                }
            }
        }
        return retVal;
    }

    /**
     * Create a training file from the first line of a data file.  All errors in here
     * are quiesced.  If we can't do it, we assume the data file is not appropriate.
     *
     * @param trainFile		training file to create
     * @param dataFile		data file whose header needs to be copied
     */
    private void createTrainingFile(File trainFile, File dataFile) {
        try (LineReader dataLines = new LineReader(dataFile)) {
            String headerLine = dataLines.next();
            try (PrintWriter writer = new PrintWriter(trainFile)) {
                writer.println(headerLine);
            }
        } catch (Exception e) {
            log.info("Error trying to create training file from data file: " + e.toString());
        }
    }

    /**
     * @return an array of metadata columns chosen by the user; the first is the ID and the last is the label column (if classifying)
     *
     * @param availableHeaders		list of available headers
     *
     * @throws IOException
     */
    private String[] findMetaColumns(List<String> availableHeaders) throws IOException {
        Stage stage = new Stage();
        String[] headers = new String[availableHeaders.size()];
        headers = availableHeaders.toArray(headers);
        MetaDialog dialog = (MetaDialog) BaseController.loadFXML(App.class, "MetaDialog", stage);
        dialog.init(headers, this.modelType);
        stage.showAndWait();
        return dialog.getResult();
    }

    /**
     * Configure according to the chosen model type.
     */
    private void configureType() {
        this.lblModelType.setText(this.modelType.getDescription());
        String caption = this.modelType.resultDescription();
        this.btnViewResults.setText(caption);
    }

    /**
     * Activate or deactivate the buttons depending on whether or not there is a valid model.
     *
     * @param b		TRUE if there is a valid model, else FALSE
     */
    private void setState(boolean b) {
        this.btnAbortCommand.setDisable(true);
        this.btnTrainingSearch.setDisable(! b);
        this.btnViewLog.setDisable(! b);
        this.btnViewResults.setDisable(! b  || ! this.modelFile.exists());
        this.btnCrossValidate.setDisable(! b);
        this.btnConvert.setDisable(! b);
        this.btnTrainingView.setDisable(! b);
        this.btnPredict.setDisable(! b);
        if (! b)
            this.lblModelType.setText("");
    }

    /**
     * Activate or deactivate the buttons depending on whether or not a background task is running.
     *
     * @param b		TRUE if there is a background task, else FALSE
     */
    private void disableButtons(boolean b) {
        this.btnAbortCommand.setDisable(! b);
        this.btnTrainingSearch.setDisable(b);
        this.btnCrossValidate.setDisable(b);
        this.btnGetDirectory.setDisable(b);
        this.btnViewResults.setDisable(b || ! this.modelFile.exists());
        this.btnConvert.setDisable(b);
    }

    /**
     * Execute a command.
     *
     * @param processor		processor on which to execute command
     * @param parms			parameters to pass
     */
    private void executeCommand(ICommand processor, String name, String[] parms) {
        // Clear the throttling counts.
        this.numSubmitted = 0;
        this.numDisplayed = 0;
        // Denote we have no score color.
        this.barColor = Color.BLACK;
        // Parse the parameters.
        boolean ok = processor.parseCommand(parms);
        if (! ok)
            BaseController.messageBox(Alert.AlertType.ERROR, "Command Error", "Invalid parameter combination.");
        else {
            // Insure the user doesn't start anything else.
            this.disableButtons(true);
            // Clear the stop flag.
            this.stopFlag = false;
            // Tell the user what we're up to.
            txtMessageBuffer.setText("Running " + name + " command.");
            // Start the thread.
            backgrounder = new BackgroundTask(this, processor);
            new Thread(backgrounder).start();
        }
    }

    /**
     * This method is called when a background task has a message to display.  It operates on
     * the background thread.
     *
     * @param message		message to display
     */
    @Override
    public void showMessage(String message) {
        Platform.runLater(new Runnable() {
            @Override
            public void run() {
                txtMessageBuffer.setText(message);
            }
        });
    }

    /**
     * This method is called when the background task has major text data to display.  It
     * operates on the background thread.
     *
     * @param paragraph		text to display
     */
    @Override
    public void showResults(String paragraph) {
        Platform.runLater(new Runnable() {
            @Override
            public void run() {
                String[] lines = LINE_END.split(paragraph);
                txtResultsPane.setText(StringUtils.join(lines, LINE_SEP));
            }
        });
    }

    @Override
    public void displayEpoch(int epoch, double score, double rating, boolean saved) throws InterruptedException {
        // Check for an abort.
        if (this.stopFlag) {
            throw new InterruptedException("Stop requested by user.");
        }
        // The throttle skips the score update unless the previous update has cleared.
        if (this.numDisplayed >= this.numSubmitted) {
            // Denote we have requested another update.
            this.numSubmitted++;
            // Submit the update.
            Platform.runLater(new Runnable() {
                @Override
                public void run() {
                    // Update the epoch and the score.
                    String epochText = Integer.toString(epoch);
                    txtEpoch.setText(epochText);
                    txtScore.setText(String.format("%14.6g", score));
                    txtRating.setText(String.format("%14.6g", rating));
                    setProgress(score);
                    // If this is the best epoch, remember it.
                    if (saved)
                        txtBestEpoch.setText(epochText);
                    // Denote we have performed the update.
                    numDisplayed++;
                }
            });
        }
    }

    @Override
    public void recordCompletion(boolean success) {
        this.backgrounder = null;
        Platform.runLater(new Runnable() {
            @Override
            public void run() {
                disableButtons(false);
                setProgress(0.0);
            }
        });
    }

    /**
     * Display the specified score value on the progress bar.
     *
     * @param score		score to display
     */
    protected void setProgress(double score) {
        double scaledScore = 0.0;
        if (score >= TrainingManager.MIN_SCORE)
            scaledScore = ((Math.log10(score) - LOG10_LOWER) * this.barContainer.getHeight() / (LOG10_UPPER - LOG10_LOWER));
        // Compute the color.
        Color color = NORMAL_COLOR;
        if (score >= 1.0)
            color = BAD_COLOR;
        else if (score >= 0.01)
            color = WARN_COLOR;
        else if (score < 1e-5)
            color = GOOD_COLOR;
        if (! this.barColor.equals(color)) {
            Background newBackground = new Background(new BackgroundFill(color, CornerRadii.EMPTY, Insets.EMPTY));
            this.barProgress.setBackground(newBackground);
        }
        this.barProgress.setPrefHeight(scaledScore);
    }

    /**
     * Convert an input file to a balanced training file.
     *
     * @param event		event descriptor
     */
    @FXML
    private void convertInput(Event event) {
        try {
            Stage convertStage = new Stage();
            BalanceInput balanceDialog = (BalanceInput) BaseController.loadFXML(App.class, "BalanceInput", convertStage);
            ITrainingProcessor processor = ModelType.create(this.modelType);
            Parms parms = new Parms(this.parmFile);
            if (! processor.setupParameters(parms, this.modelDirectory))
                throw new IOException("Invalid parameters in parameter file.");
            else {
                balanceDialog.init(this.modelDirectory, this.parmFile, processor);
                convertStage.showAndWait();
            }
        } catch (IOException e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Loading Data Converter", e.toString());
        }
    }

    /**
     * Display the training file.
     *
     * @param event		event descriptor
     */
    @FXML
    private void viewTrainingFile(Event event) {
        try {
            Stage viewStage = new Stage();
            // Get the training file.
            File trainFile = new File(this.modelDirectory, "training.tbl");
            // Extract the ID column and the meta and label columns.
            List<String> nonInputCols = this.computeNonInput();
            TrainingView viewDialog = (TrainingView) BaseController.loadFXML(App.class, "TrainingView", viewStage);
            viewDialog.init(trainFile, this.labelIdx, this.labelNames, nonInputCols, this.idColName);
            viewStage.show();
        } catch (IOException e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Loading Training File Viewer", e.toString());
        }
    }

    /**
     * This is a utility method that gets the list of non-input columns for the current model and saves the
     * name of the ID column.
     *
     * @return the list of non-input column names
     *
     * @throws IOException
     */
    private List<String> computeNonInput() throws IOException {
        ITrainingProcessor processor = ModelType.create(this.modelType);
        File parmFile = new File(this.modelDirectory, "parms.prm");
        Parms parms = new Parms(parmFile);
        if (! processor.setupParameters(parms, this.modelDirectory))
            throw new IOException("Invalid parameters in " + parmFile + ".");
        List<String> retVal = new ArrayList<String>(processor.getMetaList());
        retVal.addAll(processor.getLabelCols());
        this.idColName = processor.getIdCol();
        retVal.remove(processor.getIdCol());
        return retVal;
    }

}
