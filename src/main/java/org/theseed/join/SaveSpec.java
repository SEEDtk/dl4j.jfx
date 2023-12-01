/**
 *
 */
package org.theseed.join;

import java.awt.Desktop;
import java.io.File;
import java.io.IOException;

import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.stage.FileChooser.ExtensionFilter;

/**
 * This is the base class for all output-oriented file specifications.  It handles
 * the common controls and events.
 *
 * @author Bruce Parrello
 *
 */
public abstract class SaveSpec implements IJoinSpec {

    // FIELDS
    /** output file */
    private File outFile;
    /** parent dialog */
    private JoinDialog parent;
    /** display node for this join specification */
    private Node node;

    // CONTROLS

    /** display field for output file name */
    @FXML
    private TextField txtOutputFile;

    /** specification title */
    @FXML
    private Label lblTitle;

    /** checkbox for open-file request */
    @FXML
    private CheckBox chkOpenFile;

    @Override
    public final void init(JoinDialog parent, Node node) {
        // Connect the parent dialog.
        this.parent = parent;
        this.node = node;
        // Denote there is no output file.
        this.outFile = null;
        // Initialize the subclass.
        this.initControls();
        // If we cannot use the desktop, disable the open-file button.
        this.chkOpenFile.setVisible(Desktop.isDesktopSupported());
    }


    /**
     * Initialize the subclass controls.
     */
    protected abstract void initControls();


    /**
     * Here the user wants to specify the output file.
     *
     * @param event		event for button press
     */
    @FXML
    public void selectOutput(Event event) {
        ExtensionFilter filter = this.getFilter();
        String label = this.getFileLabel();
        File newFile = this.parent.selectOutput(filter, label);
        if (newFile != null) {
            this.outFile = newFile;
            this.txtOutputFile.setText(newFile.getName());
        }
        // Update the state of the parent.
        this.parent.configureButtons();
    }

    /**
     * @return the display label for the type of file being output
     */
    protected abstract String getFileLabel();

    /**
     * @return the filter for the type of file being output
     */
    protected abstract ExtensionFilter getFilter();

    @Override
    public final Node getNode() {
        return this.node;
    }

    /**
     * Here the user wants to delete this file spec from the file list.
     *
     * @param event		event for button press
     */
    @FXML
    public void deleteFile(Event event) {
        boolean confirmed = JoinSpec.confirmDelete(this.getOutFile());
        if (confirmed)
            this.parent.deleteFile(this);
    }

    @Override
    public final void setTitle(String title) {
        this.lblTitle.setText(title);
    }

    @Override
    public boolean isOutput() {
        return true;
    }

    /**
     * @return the output file name
     */
    public File getOutFile() {
        return outFile;
    }

    /**
     * @return the parent join dialog
     */
    protected JoinDialog getParent() {
        return this.parent;
    }

    /**
     * Open the output file if the open-file checkbox is checked.
     *
     * @throws IOException
     */
    protected void checkForOpen() throws IOException {
        if (this.chkOpenFile.isSelected()) {
            // Here the user wants to open the file in Excel.
            JoinDialog.openFile(this.outFile);
        }
    }

}
