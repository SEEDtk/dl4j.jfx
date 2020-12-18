/**
 *
 */
package org.theseed.dl4j.jfx;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;
import org.theseed.dl4j.train.ITrainReporter;
import org.theseed.dl4j.train.SearchProcessor;
import org.theseed.dl4j.train.TrainingProcessor;
import org.theseed.io.LineReader;
import org.theseed.jfx.BackgroundTask;
import org.theseed.jfx.IBackgroundController;
import org.theseed.jfx.PreferenceSet;
import org.theseed.jfx.ResizableController;
import org.theseed.utils.ICommand;

import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.geometry.Insets;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
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
    /** current model type */
    private TrainingProcessor.Type modelType;
    /** current model's parameter file */
    private File parmFile;
    /** number of epochs displayed (owned by platform thread) */
    private int numDisplayed;
    /** number of epochs queued (owned by background thread) */
    private int numSubmitted;
    /** current background task */
    private BackgroundTask backgrounder;
    /** current progress bar background color */
    private Color barColor;
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
        // Get the last-saved model directory.
        String dirName = this.getPref("modelDirectory", "");
        File newDir = null;
        if (! dirName.isEmpty())
            newDir = new File(dirName);
        try {
            boolean ok = this.analyzeModelDirectory(newDir);
            if (! ok)
                this.modelDirectory = null;
        } catch (IOException e) {
            this.setState(false);
            PreferenceSet.messageBox(Alert.AlertType.ERROR, "Error Reading Model Directory", e.getMessage());
        }
    }

    /**
     * Process a request for a new model directory.  If the directory is updated,
     * we will set up for the new model.
     *
     * @param event		event descriptor
     */
    @FXML
    private void selectModelDirectory(ActionEvent event) {
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
                        PreferenceSet.messageBox(Alert.AlertType.WARNING, "Error Changing Model Directory",
                                curDir + " does not have a training.tbl and a labels.txt.");
                    }
                } catch (IOException e) {
                    PreferenceSet.messageBox(Alert.AlertType.ERROR, "Error Changing Model Directory", e.getMessage());
                }
            }
        }
    }

    /**
     * Button event to bring up the parameter dialog and run a training search if it is successful.
     *
     * @param event		event descriptor
     */
    @FXML
    private void runSearch(ActionEvent event) {
        try {
            Stage dialogStage = new Stage();
            ParmDialog dialog = (ParmDialog) PreferenceSet.loadFXML("ParmDialog", dialogStage);
            dialog.init(this.parmFile, this.modelType);
            dialogStage.showAndWait();
            if (dialog.getResult()) {
                ICommand processor = new SearchProcessor(this);
                String[] parms = new String[] { "-t", this.modelType.toString(), this.modelDirectory.toString() };
                executeCommand(processor, "SEARCH", parms);
            }
        } catch (IOException e) {
            PreferenceSet.messageBox(Alert.AlertType.ERROR, "Error Loading Parameter Dialog", e.getMessage());
        }
    }

    /**
     * Button event to display the log view.
     *
     * @param event		event descriptor
     */
    @FXML
    private void viewLog(ActionEvent event) {
        try {
            Stage logStage = new Stage();
            LogViewer logViewer = (LogViewer) PreferenceSet.loadFXML("LogViewer", logStage);
            File logFile = new File(this.modelDirectory, "trials.log");
            logViewer.init(logFile, this.txtModelDirectory.getText());
            logStage.show();
        } catch (IOException e) {
            PreferenceSet.messageBox(Alert.AlertType.ERROR, "Error Loading Log Viewer", e.getMessage());
        }
    }

    /**
     * Button event to display the results.
     *
     * @param event		event descriptor
     */
    @FXML
    private void showModelResults(ActionEvent event) {
        if (this.modelType == TrainingProcessor.Type.CLASS) {
            // TODO show confusion matrix
        } else {
            // TODO show scatter graph
        }
    }

    /**
     * Button event to invoke cross-validation.
     *
     * @param event		event descriptor
     */
    @FXML
    private void runXValidate(ActionEvent event) {
        // TODO runXValidate
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
            this.modelDirectory = newDir;
            this.setState(false);
            this.lblModelType.setText("");
            retVal = true;
        } else {
            // Now we need to determine what type of model this is.
            File labelFile = new File(newDir, "labels.txt");
            File trainFile = new File(newDir, "training.tbl");
            if (labelFile.exists() && trainFile.exists()) {
                this.modelDirectory = newDir;
                txtModelDirectory.setText(newDir.getName());
                this.setPref("modelDirectory", newDir.getAbsolutePath());
                Set<String> labels = LineReader.readSet(labelFile);
                try (LineReader reader = new LineReader(trainFile)) {
                    String header = reader.next();
                    // If all of the labels are in this header line, it is a regression model.
                    List<String> headers = Arrays.asList(StringUtils.split(header, '\t'));
                    int count = 0;
                    for (String head : headers) {
                        if (labels.contains(head)) count++;
                    }
                    if (count == labels.size()) {
                        // Here we have a regression model.
                        modelType = TrainingProcessor.Type.REGRESSION;
                    } else {
                        modelType = TrainingProcessor.Type.CLASS;
                    }
                    configureType();
                    // Check for a parms.prm file.
                    parmFile = new File(this.modelDirectory, "parms.prm");
                    if (! parmFile.exists()) {
                        // Here we must create one.
                        TrainingProcessor processor = TrainingProcessor.create(modelType);
                        // Set the defaults.
                        processor.setAllDefaults();
                        // Count the training set.
                        int size = 0;
                        while (reader.hasNext()) {
                            reader.next();
                            size++;
                        }
                        // Compute the testing set size.
                        int testSize = size / 10;
                        if (testSize < 1) testSize = 1;
                        processor.setTestSize(testSize);
                        // Pull up the meta-column dialog.  We need to delete the label columns from the header list first.
                        List<String> availableHeaders;
                        if (modelType != TrainingProcessor.Type.REGRESSION)
                            availableHeaders = headers;
                        else
                            availableHeaders = headers.stream().filter(x -> ! labels.contains(x)).collect(Collectors.toList());
                        String[] metaCols = this.findMetaColumns(availableHeaders);
                        processor.setMetaCols(metaCols);
                        if (metaCols.length > 0)
                            processor.setIdCol(metaCols[0]);
                        processor.writeParms(parmFile);
                    }
                    // If we made it here without any errors, we can enable the buttons.
                    this.setState(true);
                    retVal = true;
                }
            }
        }
        return retVal;
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
        MetaDialog dialog = (MetaDialog) PreferenceSet.loadFXML("MetaDialog", stage);
        dialog.init(headers, this.modelType);
        stage.showAndWait();
        return dialog.getResult();
    }

    /**
     * Configure according to the chosen model type.
     */
    private void configureType() {
        this.lblModelType.setText(this.modelType.getDescription());
        String caption;
        if (this.modelType == TrainingProcessor.Type.CLASS)
            caption = "Confusion Matrix";
        else
            caption = "Scatter Graph";
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
        this.btnViewResults.setDisable(! b);
        this.btnCrossValidate.setDisable(! b);
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
        this.btnViewResults.setDisable(b);
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
            PreferenceSet.messageBox(Alert.AlertType.ERROR, "Command Error", "Invalid parameter combination.");
        else {
            // Insure the user doesn't start anything else.
            this.disableButtons(true);
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
    public void displayEpoch(int epoch, double score, boolean saved) {
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
                    txtScore.setText(String.format("%14.8g", score));
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


}
