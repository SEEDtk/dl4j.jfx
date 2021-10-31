/**
 *
 */
package org.theseed.join;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.jfx.MeanBiasDialog;
import org.theseed.io.KeyedFileMap;
import org.theseed.jfx.BaseController;
import org.theseed.jfx.ResizableController;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.TextField;
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

    // CONTROLS

    /** output file name */
    @FXML
    private TextField txtOutputFile;

    /** run button */
    @FXML
    private Button btnJoinFiles;

    /** container for join specs */
    @FXML
    private HBox joinBox;

    /** combo box for new-file type */
    @FXML
    private ChoiceBox<JoinType> cmbJoinType;

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
     *
     * @throws IOException
     */
    public void init(File parentDirectory) throws IOException {
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
        IJoinSpec initController = JoinType.INIT.getController(this);
        this.joinBox.getChildren().add(initController.getNode());
        this.specList.add(initController);
        // Set up the add-file type.
        this.cmbJoinType.getItems().addAll(JoinType.usefulValues());
        this.cmbJoinType.getSelectionModel().select(JoinType.NATURALJOIN);
        // Configure the run button.
        this.configureButtons();
        this.configureJoinBox();
    }

    /*
     * Insure the run button is only enabled if we are ready.
     */
    protected void configureButtons() {
        boolean invalid = (this.outFile == null);
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
    private void addFileSpec(ActionEvent event) {
        try {
            // Create the new file specification and add it to the file-spec lists.
            IJoinSpec newFileSpec = cmbJoinType.getSelectionModel().getSelectedItem().getController(this);
            this.joinBox.getChildren().add(newFileSpec.getNode());
            this.specList.add(newFileSpec);
            // Configure the join box width and the button state.
            this.configureJoinBox();
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
     * This event fires when the user wants to select the output file.
     *
     * @param event		button-click event
     */
    @FXML
    private void selectOutputFile(ActionEvent event) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select Join Output File");
        chooser.setInitialDirectory(this.parentDirectory);
        chooser.getExtensionFilters().addAll(MeanBiasDialog.TEXT_FILES,
                new ExtensionFilter("All Files", "*.*"));
        // Get the file.
        File retVal = chooser.showSaveDialog(this.getStage());
        if (retVal != null) {
            this.outFile = retVal;
            this.parentDirectory = retVal.getParentFile();
            this.txtOutputFile.setText(this.outFile.getName());
            this.configureButtons();
        }
    }

    /**
     * This event first when the user wants to join the files.
     *
     * @param event		button-click event
     */
    @FXML
    private void joinFiles(ActionEvent event) {
        try {
            // Get the key column name.
            String keyName = ((FileSpec) this.specList.get(0)).getKeyColumn();
            KeyedFileMap outputMap = new KeyedFileMap(keyName);
            // Run through all the file specs, applying them.
            for (IJoinSpec joinSpec : this.specList)
                joinSpec.apply(outputMap);
            // Write out the result.
            outputMap.write(this.outFile);
            // Tell the user we're done.
            BaseController.messageBox(AlertType.INFORMATION, "Join Operation Complete",
                    "Join output written to " + this.outFile + ".");
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
    }

}
