/**
 *
 */
package org.theseed.join;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.io.KeyedFileMap;

import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.Event;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableColumn.CellEditEvent;
import javafx.scene.control.TablePosition;
import javafx.scene.control.TableView;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.control.cell.TextFieldTableCell;
import javafx.util.converter.DoubleStringConverter;

/**
 * This join specification adds a new column that classifies an existing column by numeric
 * range.  The user specifies the number of classes, which sets up a table with the indicated
 * number of rows.  The first column of the table specifies the label to use and the second
 * column the maximum numeric value that gets classed with that label.  The maximum values in
 * the second column must be numeric, distinct, and ascending.
 *
 * @author Bruce Parrello
 *
 */
public class ClassifySpec implements IJoinSpec {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(ClassifySpec.class);
    /** list of class limits */
    private ObservableList<ClassLimit> limitList;
    /** parent dialog */
    private JoinDialog parent;
    /** display node for this join specification */
    private Node node;

    /**
     * This is a small object that relates a class name string to a max-value number.
     */
    public static class ClassLimit {

        private String name;
        private double maxValue;

        /**
         * Create a blank class limit.
         */
        public ClassLimit() {
            this.name = "name";
            this.maxValue = 0.0;
        }

        /**
         * Create a class limit with a specified name and maximum.
         *
         * @param className		class name
         * @param classMax		maximum value
         */
        public ClassLimit(String className, double classMax) {
            this.name = className;
            this.maxValue = classMax;
        }

        /**
         * @return the class name
         */
        public String getName() {
            return this.name;
        }

        /**
         * Specify a new class name.
         *
         * @param name 		the new class name
         */
        public void setName(String name) {
            this.name = name;
        }

        /**
         * @return the maximum value for this class
         */
        public double getMaxValue() {
            return this.maxValue;
        }

        /**
         * Specify a new maximum value.
         * @param maxValue 		the class maximum value
         */
        public void setMaxValue(double maxValue) {
            this.maxValue = maxValue;
        }

        /**
         * @return TRUE if the specified value is in this class's range, else FALSE
         *
         * @param value		value to check
         */
        public boolean isInRange(double value) {
            return (value <= this.maxValue);
        }

    }

    /**
     * This is an event handler to trigger a validation check after changing a text field.
     */
    private class ChangeHandler implements ChangeListener<String> {

        @Override
        public void changed(ObservableValue<? extends String> observable, String oldValue, String newValue) {
            ClassifySpec.this.parent.configureButtons();
        }

    }

    /**
     * This is an event handler to configure the setup-tables button after the num-classes
     * text box changes.  The text has to be an integer greater than 1.
     */
    private class NumChangeHandler implements ChangeListener<String> {

        @Override
        public void changed(ObservableValue<? extends String> observable, String oldValue, String newValue) {
            boolean invalid = newValue.isBlank();
            if (! invalid)
                invalid = ! StringUtils.containsOnly(newValue, "1234567890") ||
                        Integer.valueOf(newValue) <= 1;
            ClassifySpec.this.btnSetupTable.setDisable(invalid);
        }

    }

    /**
     * This is an event handler to prevent editing the last row's limit.  Editing is
     * automatically turned off by the "edit(-1, null)" method.
     */
    private class EditHandler implements EventHandler<CellEditEvent<ClassLimit, Double>> {

        @Override
        public void handle(CellEditEvent<ClassLimit, Double> event) {
            TablePosition<ClassLimit, Double> position = event.getTablePosition();
            if (position.getRow() == ClassifySpec.this.limitList.size() - 1)
                ClassifySpec.this.tblClasses.edit(-1, null);
        }

    }

    /**
     * This event handler updates the class limit name after an edit in the first column.
     */
    private class ChangeNameHandler implements EventHandler<TableColumn.CellEditEvent<ClassLimit, String>> {

        @Override
        public void handle(CellEditEvent<ClassLimit, String> event) {
            TablePosition<ClassLimit, String> position = event.getTablePosition();
            ClassifySpec.this.limitList.get(position.getRow()).setName(event.getNewValue());
            ClassifySpec.this.parent.configureButtons();
        }

    }

    /**
     * This event handler updates the class limit value after an edit in the second column.
     */
    private class ChangeMaxHandler implements EventHandler<TableColumn.CellEditEvent<ClassLimit, Double>> {

        @Override
        public void handle(CellEditEvent<ClassLimit, Double> event) {
            TablePosition<ClassLimit, Double> position = event.getTablePosition();
            ClassifySpec.this.limitList.get(position.getRow()).setMaxValue(event.getNewValue());
            ClassifySpec.this.parent.configureButtons();
        }

    }

    // CONTROLS

    /** title label */
    @FXML
    private Label lblTitle;

    /** name of source column to examine */
    @FXML
    private TextField txtSourceColumn;

    /** name of new column to add */
    @FXML
    private TextField txtNewColumn;

    /** number of classes to set up in the table */
    @FXML
    private TextField txtNumClasses;

    /** button to set up the table */
    @FXML
    private Button btnSetupTable;

    /** table of class labels and limits */
    @FXML
    private TableView<ClassLimit> tblClasses;

    /** class name column */
    @FXML
    private TableColumn<ClassLimit, String> colLabel;

    /** class max-value column */
    @FXML
    private TableColumn<ClassLimit, Double> colLimit;

    /** message area */
    @FXML
    private TextArea txtMessage;

    @Override
    public void init(JoinDialog parent, Node node) {
        this.parent = parent;
        this.node = node;
        // Default to two classes.
        this.txtNumClasses.setText("2");
        // Organize the table columns.  Both are editable.
        this.tblClasses.setEditable(true);
        this.colLabel.setCellValueFactory(new PropertyValueFactory<ClassLimit, String>("name"));
        this.colLabel.setCellFactory(TextFieldTableCell.<ClassLimit>forTableColumn());
        this.colLabel.setOnEditCommit(this.new ChangeNameHandler());
        this.colLimit.setCellValueFactory(new PropertyValueFactory<ClassLimit, Double>("maxValue"));
        this.colLimit.setCellFactory(TextFieldTableCell.<ClassLimit, Double>forTableColumn(new DoubleStringConverter()));
        this.colLimit.setOnEditCommit(this.new ChangeMaxHandler());
        // Start with two classes.  Note that the actual item list is a local field so we
        // can get to it easily.
        this.limitList = FXCollections.observableArrayList(new ClassLimit("Low", 0.5),
                new ClassLimit("High", Double.POSITIVE_INFINITY));
        this.tblClasses.setItems(this.limitList);
        // Insure we can't edit the limit on the last row.
        this.colLimit.setOnEditStart(this.new EditHandler());
        // Set up a listener to validate the number-of-classes text box
        this.txtNumClasses.textProperty().addListener(this.new NumChangeHandler());

        // Set up listeners to configure the parent dialog's buttons when the text box contents change.
        ChangeListener<String> handler = this.new ChangeHandler();
        this.txtNewColumn.textProperty().addListener(handler);
        this.txtSourceColumn.textProperty().addListener(handler);
    }

    @Override
    public boolean isValid() {
        // We are valid if we have two column names and the class limits are all valid.
        boolean retVal = (! this.txtNewColumn.getText().isBlank() &&
                ! this.txtSourceColumn.getText().isBlank());
        // Loop through the class limits.  Each label must be unique.
        Set<String> labels = new TreeSet<String>();
        double prev = Double.NEGATIVE_INFINITY;
        Iterator<ClassLimit> iter = this.limitList.iterator();
        while (retVal && iter.hasNext()) {
            ClassLimit limit = iter.next();
            retVal = labels.add(limit.name) && prev < limit.maxValue;
            prev = limit.maxValue;
        }
        return retVal;
    }

    /**
     * Delete this spec from the join spec list.
     *
     * @param event		event from the delete button-press
     */
    @FXML
    public void deleteFile(Event event) {
        parent.deleteFile(this);
    }

    /**
     * Update the number of classifications.
     *
     * @param event		event from the update button press
     */
    @FXML
    public void setupTable(Event event) {
        // The button can only be pressed to activate this event if the size is
        // an integer >= 2.
        int newSize = Integer.valueOf(this.txtNumClasses.getText());
        // Shrink the table until it is not too big.
        while (newSize < this.limitList.size())
            this.limitList.remove(this.limitList.size() - 2);
        // Grow the table until it is not too small.
        while (newSize > this.limitList.size())
            this.limitList.add(this.limitList.size() - 1, new ClassLimit());
        this.parent.configureButtons();
    }

    @Override
    public void apply(KeyedFileMap keyedMap) throws IOException {
        // Find the classification column.
        int classColIdx = keyedMap.findColumn(this.txtSourceColumn.getText());
        if (classColIdx < 0)
            throw new IOException("Specified source column not found.");
        // Add the header for the new column.
        keyedMap.addHeaders(Arrays.asList(this.txtNewColumn.getText()));
        // Count the number of records processed.
        int processed = 0;
        int invalid = 0;
        // Loop through them.
        for (List<String> record : keyedMap.getRecords()) {
            // Get the classification value.
            String className = "";
            String fileValue = record.get(classColIdx);
            try {
                double fileValueNum = Double.valueOf(fileValue);
                // Compute the class name from the value.
                Iterator<ClassLimit> iter = this.limitList.iterator();
                boolean found = false;
                while (iter.hasNext() && ! found) {
                    ClassLimit curr = iter.next();
                    if (curr.isInRange(fileValueNum)) {
                        found = true;
                        className = curr.getName();
                    }
                }
            } catch (NumberFormatException e) {
                // Here we have an invalid numeric, so we count it.  The class name will be empty.
                invalid++;
            }
            // Add the classification at the end of the record.
            record.add(className);
            processed++;
        }
        // Tell the user we're done.
        this.txtMessage.setText(String.format("%d records processed, %d were invalid.",
                processed, invalid));
    }

    @Override
    public Node getNode() {
        return this.node;
    }

    @Override
    public void setTitle(String title) {
        this.lblTitle.setText(title);
    }

    @Override
    public boolean isOutput() {
        return false;
    }

}
