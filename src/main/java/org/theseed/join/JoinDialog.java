/**
 *
 */
package org.theseed.join;

import java.awt.Desktop;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.io.KeyedFileMap;
import org.theseed.jfx.BaseController;
import org.theseed.jfx.ResizableController;

import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.HBox;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.FileChooser.ExtensionFilter;

/**
 * This dialog allows the user to join multiple tab-delimited files.  Its most common usage is to add meta-data from
 * other sources to a prediction file.
 *
 * @author Bruce Parrello
 *
 */
public class JoinDialog extends ResizableController {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(JoinDialog.class);
    /** output file */
    private File outFile;
    /** additional input file specifications */
    private List<IJoinSpec> specList;
    /** default directory for file dialogs */
    private File parentDirectory;
    /** file filter for all files */
    public static final ExtensionFilter ALL_FILTER = new ExtensionFilter("All Files", "*.*");
    /** file filter for excel files */
    public static final ExtensionFilter EXCEL_FILTER = new ExtensionFilter("Excel File", "*.xlsx");
    /** file filter for flat files */
    public static final ExtensionFilter FLAT_FILTER = new ExtensionFilter("Tab-Delimited File", "*.tsv", "*.txt", "*.tbl");
    /** file filter for CSV files */
    public static final ExtensionFilter CSV_FILTER = new ExtensionFilter("Comma-Delimited Files", "*.csv");

    // CONTROLS

    /** run button */
    @FXML
    private Button btnJoinFiles;

    /** container for join specs */
    @FXML
    private HBox joinBox;

    /** combo box for new-file type */
    @FXML
    private ChoiceBox<JoinType> cmbJoinType;

    /** scroll pane */
    @FXML
    private ScrollPane scrlSpecs;

    /**
     * Initialize the join dialog.
     */
    public JoinDialog() {
        super(200, 200, 800, 600);
    }

    @Override
    public String getIconName() {
        return "/org/theseed/join/join-16.png";
    }

    @Override
    public String getWindowTitle() {
        return "Join Tab-Delimited Files";
    }

    /**
     * Initialize this join dialog.  We need to create the first file specification, for the
     * primary input file. This is the one that can't be deleted.
     *
     * @param parentDirectory	directory to use for first file dialog
     * @param initFile			initial input file (or NULL if none)
     *
     * @throws IOException
     */
    public void init(File parentDirectory, File initFile) throws IOException {
        // Denote we have no output file.
        this.outFile = null;
        // Set up the initial directory.
        this.parentDirectory = parentDirectory;
        // Denote we have no additional input files.
        this.specList = new ArrayList<IJoinSpec>();
        // Set the stage.
        Stage currStage = this.getStage();
        // Note that this window is modal.
        currStage.initModality(Modality.APPLICATION_MODAL);
        // Create the join specification and store it in the join box.
        FileSpec initController = (FileSpec) JoinType.INIT.getController(this);
        this.joinBox.getChildren().add(initController.getNode());
        this.specList.add(initController);
        if (initFile != null) {
            // Here the user has a starting file for us.
            initController.setupFile(initFile);
        }
        // Set up the add-file type.
        this.cmbJoinType.getItems().addAll(JoinType.usefulValues());
        this.cmbJoinType.getSelectionModel().select(JoinType.JOIN);
        // Set up the width listener on the join box. This insures we scroll to the
        // new specification when one is added.
        this.joinBox.widthProperty().addListener((observable, oldVal, newVal) -> {
                if (oldVal.doubleValue() < newVal.doubleValue())
                    this.scrlSpecs.setHvalue(this.scrlSpecs.getHmax());
            }
        );
        // Configure the run button.
        this.configureButtons();
        this.configureJoinBox();
    }

    /*
     * Insure the run button is only enabled if we are ready.
     */
    protected void configureButtons() {
        boolean invalid = true;
        if (this.specList.size() > 0) {
            // We can only be valid if the last file spec is an output type.
            IJoinSpec lastSpec = this.specList.get(this.specList.size() - 1);
            invalid = ! lastSpec.isOutput();
        }
        Iterator<IJoinSpec> iter = this.specList.iterator();
        while (iter.hasNext() && ! invalid)
            invalid = ! iter.next().isValid();
        this.btnJoinFiles.setDisable(invalid);
    }

    /**
     * This event fires when the user wants to add a new file.
     *
     * @param event		button-click event
     */
    @FXML
    private void addFileSpec(Event event) {
        try {
            // Create the new file specification and add it to the file-spec lists.
            IJoinSpec newFileSpec = cmbJoinType.getSelectionModel().getSelectedItem().getController(this);
            this.joinBox.getChildren().add(newFileSpec.getNode());
            this.specList.add(newFileSpec);
            // Configure the join box width.
            this.configureJoinBox();
            // Update the button states.
            this.configureButtons();
        } catch (Exception e) {
            BaseController.messageBox(AlertType.ERROR, "Error Processing Input File", e.toString());
        }
    }

    /**
     * Insure the join box is the correct width.
     */
    private void configureJoinBox() {
        this.joinBox.setPrefWidth(400.0 * this.specList.size());
    }

    /**
     * This event first when the user wants to join the files.
     *
     * @param event		button-click event
     */
    @FXML
    private void joinFiles(Event event) {
        try {
            // Get the key column name.
            String keyName = ((FileSpec) this.specList.get(0)).getKeyColumn();
            KeyedFileMap outputMap = new KeyedFileMap(keyName);
            // Run through all the file specs, applying them.
            for (IJoinSpec joinSpec : this.specList)
                joinSpec.apply(outputMap);
            // Tell the user we're done.
            BaseController.messageBox(AlertType.INFORMATION, "Join Operation Complete",
                    String.format("%d file operations performed.  %d data records in result.",
                            this.specList.size(), outputMap.size()));
        } catch (Exception e) {
            BaseController.messageBox(AlertType.ERROR, "Error in Join Operation", e.toString());
        }
    }

    /**
     * @return the parent directory for the next file operation
     */
    protected File getParentDirectory() {
        return this.parentDirectory;
    }

    /**
     * @param parentDirectory specify the parent directory for the next file operation
     */
    protected void setParentDirectory(File parentDirectory) {
        this.parentDirectory = parentDirectory;
    }

    /**
     * Remove a file from the file input list.
     *
     * @param fileSpec	join specification for file to remove
     */
    public void deleteFile(IJoinSpec fileSpec) {
        // Remove the join specification from the data structures.
        this.joinBox.getChildren().remove(fileSpec.getNode());
        this.specList.remove(fileSpec);
        // Insure the join box is the correct size.
        this.configureJoinBox();
        // Configure the join button.
        this.configureButtons();
    }

    /**
     * @return an input file chosen by the user
     *
     * @param parentFile	directory to start in
     * @param stage			stage on which controls are being displayed
     */
    public File chooseFile(ExtensionFilter filter, String label) {
        // Initialize the chooser dialog.
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select " + label);
        chooser.setInitialDirectory(this.parentDirectory);
        chooser.getExtensionFilters().addAll(filter, ALL_FILTER);
        // Get the file.
        File retVal = chooser.showOpenDialog(this.getStage());
        return retVal;
    }

    /**
     * @return a computed name for this operation
     */
    public String getName() {
        String retVal = "joined_table";
        if (this.outFile != null) {
            String name = StringUtils.substringBeforeLast(this.outFile.getName(), ".");
            retVal = name.replaceAll("[^A-Za-z0-9]", "_");
        }
        return retVal;
    }

    /**
     * Ask the user for an output file.
     *
     * @param filter		extension filter for the file type
     * @param label			label for the file type
     *
     * @return the new file, or NULL if the user cancelled out
     */
    public File selectOutput(ExtensionFilter filter, String label) {
        // Initialize the chooser dialog.
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select Output " + label);
        chooser.setInitialDirectory(this.getParentDirectory());
        chooser.getExtensionFilters().addAll(filter);
        // Get the file.
        File retVal = chooser.showSaveDialog(this.getStage());
        if (retVal != null) {
            // We have a new output file.  Save the parent directory for next time.
            this.setParentDirectory(retVal.getParentFile());
        }
        return retVal;
    }

    /**
     * Open the specified file in its default application.
     *
     * @param targetFile	file to open
     *
     * @throws IOException
     */
    public static void openFile(File targetFile) throws IOException {
        Desktop myDesktop = Desktop.getDesktop();
        myDesktop.open(targetFile);
    }

}
