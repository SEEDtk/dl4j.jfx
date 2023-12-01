/**
 *
 */
package org.theseed.join;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.io.KeyedFileMap;
import org.theseed.utils.IDescribable;

import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;

/**
 * This specification allows the user to specify a regular expression pattern.  The
 * records are either excluded or included depending on whether or not they match the pattern.
 *
 * @author Bruce Parrello
 *
 */
public class MatchSpec implements IJoinSpec {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(MatchSpec.class);
    /** compiled pattern */
    private Pattern matchPattern;
    /** parent dialog */
    private JoinDialog parent;
    /** display node for this join specification */
    private Node node;

    /**
     * Enum for the match modes
     */
    public static enum Mode implements IDescribable {
        INCLUDE {

            @Override
            public boolean keep(String field, Pattern pattern) {
                return pattern.matcher(field).matches();
            }

            @Override
            public String getDescription() {
                return "Include records with a matching field.";
            }
        }, EXCLUDE {

            @Override
            public boolean keep(String field, Pattern pattern) {
                return ! pattern.matcher(field).matches();
            }

            @Override
            public String getDescription() {
                return "Include only records without a matching field.";
            }
        }, SUBSTRING {

            @Override
            public boolean keep(String field, Pattern pattern) {
                return pattern.matcher(field).find();
            }

            @Override
            public String getDescription() {
                return "Include records with a matching substring in the field.";
            }
        };

        /**
         * @return TRUE if we should keep the current line, else FALSE
         *
         * @param field		value of the target field
         * @param pattern	pattern to match against it
         */
        public abstract boolean keep(String field, Pattern pattern);
    }

    /**
     * This is an event handler to trigger a validation check after changing the column field.
     */
    private class ChangeHandler implements ChangeListener<String> {

        @Override
        public void changed(ObservableValue<? extends String> observable, String oldValue, String newValue) {
            MatchSpec.this.parent.configureButtons();
        }

    }

    /**
     * This is an event handler to trigger a pattern update after losing focus on the
     * pattern text-area.
     */
    private class FocusHandler implements ChangeListener<Boolean> {

        @Override
        public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
            MatchSpec.this.updatePattern();
        }

    }

    /**
     * This is an event handler that displays a description when the mode changes.
     */
    private class ModeHandler implements ChangeListener<Mode> {

        @Override
        public void changed(ObservableValue<? extends Mode> observable, Mode oldValue, Mode newValue) {
            MatchSpec.this.txtMessage.setText(newValue.getDescription());
        }

    }

    // CONTROLS

    /** label containing the spec title */
    @FXML
    private Label lblTitle;

    /** name of the column for matching */
    @FXML
    private TextField txtMatchColumn;

    /** regular expression for pattern matching */
    @FXML
    private TextArea txtPattern;

    /** checked for case-insensitive matching */
    @FXML
    private CheckBox chkInsensitive;

    /** checked for literal matching (no regex parsing) */
    @FXML
    private CheckBox chkLiteral;

    /** mode for matching */
    @FXML
    private ChoiceBox<Mode> cmbMode;

    /** error message area */
    @FXML
    private TextArea txtMessage;


    @Override
    public void init(JoinDialog parent, Node node) {
        this.parent = parent;
        this.node = node;
        // Denote we have no match pattern.
        this.matchPattern = null;
        // Set up the events for the match column name and the pattern field.
        this.txtPattern.focusedProperty().addListener(this.new FocusHandler());
        this.txtMatchColumn.textProperty().addListener(this.new ChangeHandler());
        // Store the modes in the mode box.
        this.cmbMode.setItems(FXCollections.observableArrayList(Mode.values()));
        this.cmbMode.getSelectionModel().clearAndSelect(0);
        // Set a handler to display mode changes.
        this.cmbMode.valueProperty().addListener(this.new ModeHandler());
    }

    @Override
    public boolean isValid() {
        return (! this.txtMatchColumn.getText().isBlank() && this.matchPattern != null);
    }

    @Override
    public void apply(KeyedFileMap keyedMap) throws IOException {
        // Find the match column.
        int keyColIdx = keyedMap.findColumn(this.txtMatchColumn.getText());
        if (keyColIdx < 0)
            throw new IOException("Could not find Match Column.");
        // Count the records processed and kept.
        int processed = 0;
        int kept = 0;
        // Loop through the records, removing the appropriate ones.
        Mode mode = this.cmbMode.getValue();
        Iterator<List<String>> iter = keyedMap.getRecords().iterator();
        while (iter.hasNext()) {
            List<String> record = iter.next();
            String field = record.get(keyColIdx);
            boolean keep = mode.keep(field, this.matchPattern);
            if (! keep)
                iter.remove();
            else
                kept++;
            processed++;
        }
        // Describe the result.
        this.txtMessage.setText(String.format("%d records processed, %d kept.", processed, kept));
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

    /**
     * This event is triggered when one of the check boxes is changed.
     *
     * @event	action event for the button press
     */
    @FXML
    private void updatePattern(Event event) {
        this.updatePattern();
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
     * This is a utility method that updates the match pattern from the pattern text
     * and the check boxes.  If the pattern fails to compile, a NULL will be stored.
     */
    protected void updatePattern() {
        try {
            // Compute the match flags.
            int flags = 0;
            if (this.chkInsensitive.isSelected())
                flags += Pattern.CASE_INSENSITIVE;
            if (this.chkLiteral.isSelected())
                flags += Pattern.LITERAL;
            // Compile the match pattern.
            this.matchPattern = Pattern.compile(this.txtPattern.getText(), flags);
            // Here we succeeded, so blank the text message.
            this.txtMessage.setText("");
        } catch (PatternSyntaxException e) {
            // Display the error and null out the pattern.
            this.txtMessage.setText(e.getMessage());
            this.matchPattern = null;
        }
        // Configure the buttons in the parent dialog.
        this.parent.configureButtons();
    }
}
