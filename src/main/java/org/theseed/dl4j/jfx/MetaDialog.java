/**
 *
 */
package org.theseed.dl4j.jfx;

import java.util.ArrayList;
import java.util.List;

import org.theseed.dl4j.train.ModelType;
import org.theseed.jfx.MovableController;

import javafx.collections.ObservableList;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;

/**
 * This is the dialog for selecting the metadata columns.  It takes as input a list of column headers and places them in the
 * left list box.  Clicking on a list entry transfers it to the other box.  When the DONE button is clicked, the selected
 * headers are passed back to the caller.
 *
 * @author Bruce Parrello
 *
 */
public class MetaDialog extends MovableController {

    // FIELDS
    /** basic instructions */
    private static String INSTRUCTIONS = "Click on a column name to move it between list boxes.  The metadata columns should be listed on the right "
            + "when you are done.  The first column selected will be used as the row ID.";
    // CONTROLS

    /** list of unselected columns */
    @FXML
    private ListView<String> leftList;

    /** list of selected columns */
    @FXML
    private ListView<String> rightList;

    /** instructional text label */
    @FXML
    private Label instructions;

    public MetaDialog() {
        super(200, 200);
    }

    @Override
    public String getIconName() {
        return "fig-gear-16.png";
    }

    @Override
    public String getWindowTitle() {
        return "Metadata Column Selection";
    }

    /**
     * Initialize this dialog.
     *
     * @param headers		column header names
     * @param modelType		relevant model type (CLASS or REGRESSION)
     */
    public void init(String[] headers, ModelType modelType) {
        // Start out with everything unselected.
        leftList.getItems().addAll(headers);
        // Fix the instructions.
        if (modelType != ModelType.REGRESSION) {
            instructions.setText(INSTRUCTIONS + " The last column selected will be the label (this is required).");
        } else {
            instructions.setText(INSTRUCTIONS);
        }
    }

    /**
     * Handle a click on the left list.  All the selected items will be moved to the right list.
     *
     * @param event			mouse event descriptor
     */
    @FXML
    private void clickLeft(Event event) {
        this.transferSelected(this.leftList, this.rightList);
    }

    /**
     * Handle a click on the right list.  All the selected items will be moved to the left list.
     *
     * @param event			mouse event descriptor
     */
    @FXML
    private void clickRight(Event event) {
        this.transferSelected(this.rightList, this.leftList);
    }

    /**
     * Move the selected items in the source list to the target list
     *
     * @param fromList		source list
     * @param toList		target list
     */
    private void transferSelected(ListView<String> fromList, ListView<String> toList) {
        // Get the selected items in the from list.  We make a copy so that we don't worry
        // about it changing under us.
        List<String> selected = new ArrayList<String>(fromList.getSelectionModel().getSelectedItems());
        // Get all the items in each list.
        ObservableList<String> target = toList.getItems();
        ObservableList<String> source = fromList.getItems();
        for (String item : selected) {
            source.remove(item);
            target.add(item);
        }
    }

    /**
     * Close the window.
     *
     * @param event		button click event descriptor
     */
    @FXML
    private void finish(Event event) {
        this.close();
    }

    /**
     * @return the array of selected column headers
     */
    public String[] getResult() {
        return this.rightList.getItems().stream().toArray(String[]::new);
    }

}
