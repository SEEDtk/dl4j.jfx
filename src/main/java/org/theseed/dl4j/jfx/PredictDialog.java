/**
 *
 */
package org.theseed.dl4j.jfx;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.train.ModelType;
import org.theseed.io.LineReader;
import org.theseed.io.TabbedLineReader;
import org.theseed.jfx.BaseController;
import org.theseed.jfx.MovableController;
import org.theseed.join.JoinDialog;
import org.theseed.utils.Parms;

import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.TextField;
import javafx.stage.FileChooser;
import javafx.stage.FileChooser.ExtensionFilter;

/**
 * This dialog accepts an input file without expectations and applies the model to produce a
 * prediction file.  This is a convenient way to apply the model from the GUI.
 *
 * @author Bruce Parrello
 *
 */
public class PredictDialog extends MovableController {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(PredictDialog.class);
    /** controlling model directory */
    private File modelDir;
    /** controlling model type */
    private ModelType modelType;
    /** saved directory for input file */
    private File savedDirectory;
    /** input file */
    private File inputFile;
    /** output file */
    private File outputFile;
    /** set of input column headers in the training file */
    private Set<String> inputColumns;
    /** list of meta-data column headers in the input file */
    private List<String> metaColumns;

    // CONTROLS

    /** text field for displaying input file name */
    @FXML
    private TextField txtInputFile;

    /** text field for displaying output file name */
    @FXML
    private TextField txtOutputFile;

    /** button to start the process */
    @FXML
    private Button btnRun;

    /** checkbox to request join dialog */
    @FXML
    private CheckBox chkJoinRequest;



    public PredictDialog() {
        super(400, 400);
    }

    @Override
    public String getIconName() {
        return "job-16.png";
    }

    @Override
    public String getWindowTitle() {
        return "Make Predictions";
    }

    /**
     * Initialize this window with information from the model directory.
     *
     * @param modelDirectory	directory for the model to apply
     * @param modelType 		type of the model
     */
    public void init(File modelDirectory, ModelType modelType) {
        // Denote we have no input or output file.
        this.inputFile = null;
        this.outputFile = null;
        this.txtInputFile.setText("");
        this.txtOutputFile.setText("");
        // Save the model directory and type.
        this.modelDir = modelDirectory;
        this.modelType = modelType;
        this.savedDirectory = modelDirectory;
        try {
            // The next step is to get the list of input columns.  We use the input column list
            // to compute the meta-data columns when a training file is loaded, as well as to validate
            // the file.
            File trainFile = new File(modelDirectory, "training.tbl");
            File labelFile = new File(modelDirectory, "labels.txt");
            File parmFile = new File(modelDirectory, "parms.prm");
            // First, we get the names of the metadata columns in the training file.
            Parms parms = new Parms(parmFile);
            String metaString = parms.getValue("--meta");
            Set<String> metaCols = Arrays.stream(StringUtils.split(metaString, ",")).collect(Collectors.toSet());
            if (modelType.metaLabel() > 0) {
                // Here we are a classification model.  The column parameter is interrogated.
                String labelCol = parms.getValue("--col");
                if (labelCol.isEmpty())
                    throw new IOException("Model is classification-based, but has no identifier label column.");
                metaCols.add(labelCol);
            } else {
                // Here we are a regression model.  Add the labels to the meta-column set.
                Set<String> labelCols = LineReader.readSet(labelFile);
                metaCols.addAll(labelCols);
            }
            // Now we get all of the columns from the training file that are NOT in the meta-column set.
            // These are the input columns.
            try (TabbedLineReader trainStream = new TabbedLineReader(trainFile)) {
                this.inputColumns = new HashSet<String>(trainStream.size() * 3 / 2);
                for (String header : trainStream.getLabels()) {
                    if (! metaCols.contains(header))
                        this.inputColumns.add(header);
                }
            }
            log.info("{} input columns found in training file.", this.inputColumns.size());
            // Set up the buttons.
            this.configureButtons();
        } catch (Exception e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Computing Predictions", e.toString());
        }
    }

    @FXML
    private void selectInput(Event event) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select Input File");
        chooser.setInitialDirectory(this.savedDirectory);
        chooser.getExtensionFilters().addAll(JoinDialog.FLAT_FILTER,
                new ExtensionFilter("All Files", "*.*"));
        // Get the file.
        File retVal = chooser.showOpenDialog(this.getStage());
        if (retVal != null) {
            // Save the parent directory no matter what.
            this.savedDirectory = retVal.getParentFile();
            // Validate the input file.
            if (! this.checkInputFile(retVal))
                this.inputFile = null;
            else {
                this.inputFile = retVal;
                this.txtInputFile.setText(this.inputFile.getName());
            }
            this.configureButtons();
        }
    }

    /**
     * Compute the metadata columns for this input file and verify that it is valid.
     *
     * @param inFile	input file name
     *
     * @return TRUE if successful, FALSE if the file is invalid
     */
    private boolean checkInputFile(File inFile) {
        boolean retVal = false;
        try (TabbedLineReader inStream = new TabbedLineReader(inFile)) {
            // Get the set of all columns in the file.
            Set<String> fileCols = Arrays.stream(inStream.getLabels()).collect(Collectors.toSet());
            // Verify that all the required columns are present.
            if (! fileCols.containsAll(this.inputColumns))
                throw new IOException("Proposed input file is missing one or more required input columns.");
            else {
                // Here we have everything.  The columns that are NOT required are metadata columns.
                fileCols.removeAll(this.inputColumns);
                this.metaColumns = new ArrayList<String>(fileCols);
                log.info("{} metadata columns found in input.", this.metaColumns.size());
                retVal = true;
            }
        } catch (Exception e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Validating Input", e.toString());
        }
        return retVal;
    }

    @FXML
    private void selectOutput(Event event) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select Prediction Output File");
        chooser.setInitialDirectory(this.savedDirectory);
        chooser.getExtensionFilters().addAll(JoinDialog.FLAT_FILTER,
                new ExtensionFilter("All Files", "*.*"));
        // Get the file.
        File retVal = chooser.showSaveDialog(this.getStage());
        if (retVal != null) {
            this.outputFile = retVal;
            this.txtOutputFile.setText(this.outputFile.getName());
            this.configureButtons();
        }
    }

    @FXML
    private void runPredictions(Event event) {
        try {
            // Run the predictions.
            modelType.predict(modelDir, inputFile, outputFile, metaColumns);
            // Here we were successful.
            BaseController.messageBox(Alert.AlertType.INFORMATION, "Predictions", "Predictions written to "
                    + this.outputFile.getAbsolutePath() + ".");
            // Dismiss the dialog.
            this.close();
        } catch(Exception e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Prediction Error", e.toString());
        }
    }

    /**
     * Turn on the run button iff we have an input file and an output file.
     */
    private void configureButtons() {
        this.btnRun.setDisable(this.inputFile == null || this.outputFile == null);
    }

    /**
     * @return the output file if the user wants to pass it to a join dialog, else NULL
     */
    public File isJoinRequested() {
        File retVal = null;
        if (this.chkJoinRequest.isSelected())
            retVal = this.outputFile;
        return retVal;
    }

}
