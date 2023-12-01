/**
 *
 */
package org.theseed.join;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.io.KeyedFileMap;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;

/**
 * This specification allows the user to choose a random subset of the input file.
 *
 * @author Bruce Parrello
 *
 */
public class PickSpec implements IJoinSpec {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(PickSpec.class);
    /** parent dialog */
    private JoinDialog parent;
    /** display node for this join specification */
    private Node node;

    /**
     * This is an event handler to trigger a pattern update after losing focus on the
     * text control.  It triggers validation on the controls.
     */
    private class FocusHandler implements ChangeListener<Boolean> {

        @Override
        public void changed(ObservableValue<? extends Boolean> observable, Boolean oldValue, Boolean newValue) {
            PickSpec.this.parent.configureButtons();
        }

    }

    // CONTROLS

    /** label containing the spec title */
    @FXML
    private Label lblTitle;

    /** number of records to pick */
    @FXML
    private TextField txtPickNum;

    /** column to insure has unique values */
    @FXML
    private TextField txtScatterName;

    @Override
    public void init(JoinDialog parent, Node node) {
        this.parent = parent;
        this.node = node;
        // Set up the focus handler on the pick number.
        this.txtPickNum.focusedProperty().addListener(this.new FocusHandler());
    }

    @Override
    public boolean isValid() {
        // We are valid if we have a numeric positive pick count.
        boolean retVal = false;
        try {
            int pickCount = Integer.valueOf(txtPickNum.getText());
            retVal = (pickCount > 0);
        } catch (NumberFormatException e) { }
        return retVal;
    }

    @Override
    public void apply(KeyedFileMap keyedMap) throws IOException {
        // Get the number of records to keep.
        int pickNum = Integer.valueOf(this.txtPickNum.getText());
        // We will build our new file in here and then copy it over.
        List<String> headers = keyedMap.getHeaders();
        // Our strategy is to create a set for the keys we are going to keep.
        KeyedFileMap subset = new KeyedFileMap(headers);
        // Get the input file keys in random order.
        List<String> keys = new ArrayList<String>(keyedMap.getKeys());
        Collections.shuffle(keys);
        // Compute the index of the scatter column.  If we use column 0, that means there will
        // be no scattering, since column 0 is the key and every key is already unique.
        String scatterName = txtScatterName.getText();
        int scatterCol = 0;
        if (! StringUtils.isBlank(scatterName))
            scatterCol = keyedMap.findColumn(scatterName);
        if (scatterCol < 0)
            throw new IOException("Invalid scatter column name \"" + scatterName + "\".");
        // Finally, this will track the scatter values already used.
        Set<String> used = new HashSet<String>(pickNum * 4 / 3);
        // Now loop through the randomized keys, selecting records until we find enough.
        Iterator<String> iter = keys.iterator();
        while (iter.hasNext() && subset.size() < pickNum) {
            String key = iter.next();
            List<String> record = keyedMap.getRecord(key);
            if (record != null) {
                String dataCol = record.get(scatterCol);
                if (! used.contains(dataCol)) {
                    subset.addRecord(record);
                    used.add(dataCol);
                }
            }
        }
        // All done! Copy the subset into our output map.
        keyedMap.shallowCopyFrom(subset);
    }

    @Override
    public Node getNode() {
        return this.node;
    }

    @Override
    public void setTitle(String title) {
        this.lblTitle.setText("Choose Random Subset");
    }

    @Override
    public boolean isOutput() {
        return false;
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


}
