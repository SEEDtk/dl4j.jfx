/**
 *
 */
package org.theseed.dl4j.jfx;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

import org.theseed.dl4j.train.ITrainingProcessor;
import org.theseed.jfx.BaseController;
import org.theseed.jfx.Prediction;
import org.theseed.jfx.ResizableController;
import org.theseed.jfx.Stat;
import org.theseed.reports.IValidationReport;

import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Node;
import javafx.scene.control.Alert;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.AnchorPane;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;

/**
 * This is the base class for windows that display model testing results.  The actual display depends on the
 * type of processor, and is determined by the subclass.  The root node of the window scene should be a
 * Pane whose first node will be the parent of the controls used by this class.
 *
 * @author Bruce Parrello
 *
 */
public class ResultDisplay extends ResizableController {

    // FIELDS
    /** current model processor */
    private ITrainingProcessor processor;
    /** current training file */
    private File trainingFile;
    /** results display controller */
    private ValidationDisplayReport displayController;

    // CONTROLS

    /** training file name */
    @FXML
    private TextField txtTrainingFileName;

    /** client area for display control */
    @FXML
    private AnchorPane clientPane;

    /** table for prediction statistics */
    @FXML
    private TableView<Stat> statsTable;

    /** table for outliers (regression only) */
    @FXML
    private TableView<Prediction> outlierTable;

    /**
     * The constructor positions the window.
     */
    public ResultDisplay() {
        super(200, 200, 700, 500);
    }

    @Override
    public String getIconName() {
        return "session-16.png";
    }

    @Override
    public String getWindowTitle() {
        return "Result Display";
    }

    /**
     * Initialize this object.
     *
     * @param processor		training processor whose results are desired
     * @param displayPane	name of the FXML resource for the display pane
     *
     * @throws IOException
     */
    public void init(ITrainingProcessor processor, String displayPane) throws IOException {
        // Get the stage.
        Stage currStage = this.getStage();
        // Save the processor and get the model directory.
        this.processor = processor;
        File modelDir = processor.getModelDir();
        // Note that this window is modal.
        currStage.initModality(Modality.APPLICATION_MODAL);
        // Update the title with the model name.
        currStage.setTitle("Prediction Test for " + modelDir.getName());
        // Get and display the default training file.
        storeTrainingFile(new File(modelDir, "training.tbl"));
        // Get the list of labels.
        List<String> labels = processor.getLabels();
        // Load the display controls and anchor them to the display pane.
        FXMLLoader fxmlLoader = new FXMLLoader(App.class.getResource(displayPane + ".fxml"));
        Node displayNode = fxmlLoader.load();
        this.clientPane.getChildren().add(displayNode);
        AnchorPane.setLeftAnchor(displayNode, 0.0);
        AnchorPane.setRightAnchor(displayNode, 0.0);
        AnchorPane.setTopAnchor(displayNode, 0.0);
        AnchorPane.setBottomAnchor(displayNode, 0.0);
        // Initialize the controller.
        this.displayController = (ValidationDisplayReport) fxmlLoader.getController();
        this.displayController.init(labels);
        this.displayController.registerStatsTable(this.statsTable);
        this.displayController.registerOutlierTable(this.outlierTable);
        // Set up the rules for the table display.
        Stat.setupTable(this.statsTable);
        // Display the results.
        this.showResults();
    }

    /**
     * Update the training file.
     *
     * @param trainFile		new training file to use
     */
    private void storeTrainingFile(File trainFile) {
        this.trainingFile = trainFile;
        // Get the full name.
        String displayName = trainFile.getAbsolutePath();
        Tooltip t = new Tooltip(displayName);
        // Assemble the last two pieces of the name.
        String name = trainFile.getName();
        File parent = trainFile.getParentFile();
        if (parent != null)
            displayName = parent.getName() + File.separator + name;
        // Store the short name in the text field and apply a tooltip.
        this.txtTrainingFileName.setText(displayName);
        Tooltip.install(this.txtTrainingFileName, t);
    }

    /**
     * Button event to get a new training file.
     *
     * @param event		event descriptor
     */
    @FXML
    private void selectTrainingFile(Event event) {
        FileChooser chooser = createFileChooser("Select Training File");
        File newFile = chooser.showOpenDialog(this.getStage());
        if (newFile != null) {
            if (! newFile.canRead())
                BaseController.messageBox(Alert.AlertType.ERROR, "Training File Error", "File cannot be read.");
            else
                this.storeTrainingFile(newFile);
        }
    }

    /**
     * @return a file chooser for training files with the specified title
     *
     * @param title		title to use
     */
    public FileChooser createFileChooser(String title) {
        FileChooser retVal = new FileChooser();
        retVal.setTitle(title);
        retVal.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Text Files", "*.txt", "*.tbl", "*.tsv"),
                new FileChooser.ExtensionFilter("All Files", "*.*"));
        retVal.setInitialDirectory(this.trainingFile.getParentFile());
        return retVal;
    }

    /**
     * Button event to display results.
     *
     * @param event		event descriptor
     */
    @FXML
    private void displayResults(Event event) {
        this.showResults();
    }

    /**
     * Run the predictions and display the results.
     */
    private void showResults() {
        try {
            // Display the predictions.
            processor.runPredictions(this.displayController, this.trainingFile);
        } catch (Exception e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "PredictionError", e.toString());
        }
    }

    /**
     * Run the predictions and save the results to a file.
     */
    @FXML
    private void saveResults(Event event) {
        FileChooser chooser = this.createFileChooser("Select Output File");
        File outFile = chooser.showSaveDialog(this.getStage());
        if (outFile != null) {
            try (FileOutputStream outStream = new FileOutputStream(outFile)) {
                IValidationReport output = processor.getValidationReporter(outStream);
                processor.runPredictions(output, this.trainingFile);
            } catch (IOException e) {
                BaseController.messageBox(Alert.AlertType.ERROR, "PredictionError", e.toString());
            }
        }
    }

}
