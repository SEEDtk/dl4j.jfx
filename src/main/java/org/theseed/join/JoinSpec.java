/**
 *
 */
package org.theseed.join;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.io.KeyedFileMap;
import org.theseed.io.TabbedLineReader;
import org.theseed.jfx.BaseController;
import javafx.collections.ObservableList;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.MultipleSelectionModel;
import javafx.scene.control.SelectionMode;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;

/**
 * This control group represents the data for a single file in a join operation.  It allows specification of
 * the file name, the key column, the columns to keep, and a choice box indicating whether the join is
 * left-oriented or natural.  When a file is first specified, all the columns will be pre-selected.
 *
 * @author Bruce Parrello
 *
 */
public class JoinSpec implements IJoinSpec {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(JoinSpec.class);
    /** input file */
    private File inFile;
    /** parent dialog */
    private JoinDialog parent;
    /** display node for this join specification */
    private Node node;
    /** TRUE if the current file has too many columns  */
    private boolean largeFileMode;
    /** original array of headers from the input file */
    private String[] headers;
    /** maximum number of columns for normal processing */
    private static final int MAX_COLUMNS = 300;

    // CONTROLS

    /** input file name */
    @FXML
    private TextField txtInputFile;

    /** key column selector */
    @FXML
    private ChoiceBox<String> cmbKeyColumn;

    /** blank-key toggle */
    @FXML
    private CheckBox chkIgnoreEmpty;

    /** list view for selecting columns to include */
    @FXML
    private ListView<String> lstColumns;

    /** button to delete the file spec */
    @FXML
    protected Button btnDelete;

    /** specification title */
    @FXML
    private Label lblTitle;

    /** column qualifier */
    @FXML
    protected TextField txtQualifier;

    /** combo box for join type */
    @FXML
    protected ChoiceBox<Type> cmbType;

    /** button to select all columns */
    @FXML
    private Button allButton;

    /** button to select no columns */
    @FXML
    private Button noneButton;

    /** button to invert the column selection */
    @FXML
    private Button flipButton;

    /** output area for messages */
    @FXML
    protected TextArea txtMessage;

    public static enum Type {
        NATURALJOIN {
            @Override
            public String toString() {
                return "Natural";
            }

            @Override
            protected void processMissing(Iterator<Map.Entry<String, List<String>>> iter, List<String> data, int size) {
                // For a natural join, missing values are removed from the output.
                iter.remove();
            }

            @Override
            protected void process(JoinSpec joinSpec, KeyedFileMap outputMap) throws IOException {
                joinSpec.processJoin(outputMap);
            }

        }, LEFTJOIN {
            @Override
            public String toString() {
                return "Left";
            }

            @Override
            protected void processMissing(Iterator<Map.Entry<String, List<String>>> iter, List<String> data, int size) {
                // For a left join, the missing key's new fields are filled with empty strings.
                for (int i = 0; i < size; i++)
                    data.add("");
            }

            @Override
            protected void process(JoinSpec joinSpec, KeyedFileMap outputMap) throws IOException {
                joinSpec.processJoin(outputMap);
            }

        }, LINKFILES {

            @Override
            public String toString() {
                return "Link";
            }

            @Override
            protected void processMissing(Iterator<Entry<String, List<String>>> iter, List<String> data, int size) {
                // This is currently not used by the link process.
            }

            @Override
            protected void process(JoinSpec joinSpec, KeyedFileMap outputMap) throws IOException {
                joinSpec.processLink(outputMap);
            }

        };

        /**
         * @return the string representation of this join type
         */
        public abstract String toString();

        /**
         * Process a missing line for a real join.  The line's key is in the incoming output
         * map, but not the new file's input map.
         *
         * @param iter		iterator through the output map
         * @param data		current record in the output map
         * @param size		width of the new file's data
         */
        protected abstract void processMissing(Iterator<Map.Entry<String, List<String>>> iter, List<String> data, int size);

        /**
         * Perform the join.
         *
         * @param joinSpec		parent join spec that will execute the join
         * @param keyedFileMap	incoming file map
         *
         * @throws IOException
         */
        protected abstract void process(JoinSpec joinSpec, KeyedFileMap outputMap) throws IOException;
    }

   /**
     * Create a standard join-specification control.
     */
    public JoinSpec() {
    }

    /**
     * Initialize this specifier for a specific file.
     *
     * @param parent	master dialog for all the join specs
     * @param node		display node for this controller
     *
     * @throws IOException
     */
    @Override
    public void init(JoinDialog parent, Node node) {
        // Remember the parent dialog.
        this.parent = parent;
        this.node = node;
        // Clear the input file.
        this.inFile = null;
        // Perform any additional subclass initialization.
        this.initialConfigure();
        // Allow multiple selections in the column list.
        this.lstColumns.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);
    }

    /**
     * Perform any special setup for the subclass.  For this object, we set up the join-type
     * combo box.
     */
    protected void initialConfigure() {
        this.cmbType.getItems().addAll(Type.values());
        this.cmbType.getSelectionModel().clearAndSelect(0);
    }

    /**
     * Here the user has selected a file as the input file.  We update the
     * file name display and fill in the list box and the choice box.
     *
     * @param inputFile		new input file to use
     *
     * @throws IOException
     */
    public void setupFile(File inputFile) throws IOException {
        // Remember this directory.
        this.parent.setParentDirectory(inputFile.getParentFile());
        // Erase the current control contents.
        ObservableList<String> keyColumn = this.cmbKeyColumn.getItems();
        ObservableList<String> dataColumns = this.lstColumns.getItems();
        keyColumn.clear();
        dataColumns.clear();
        // Get the column headers for this file and fill the controls.
        try (TabbedLineReader inStream = new TabbedLineReader(inputFile)) {
            if (inStream.size() < 1)
                throw new IOException("Input file must contain at least one column.");
            // Save the file headers.
            this.headers = inStream.getLabels();
            if (inStream.size() > MAX_COLUMNS) {
                // Here we have a large file situation.  The display controls will lock if we
                // try to put all the columns in them.  We force the key column to be the
                // first, and just auto-select all the data columns for processing.
                this.largeFileMode = true;
                keyColumn.add(inStream.getLabels()[0]);
                BaseController.messageBox(AlertType.WARNING, "Large Input File",
                        String.format("This file has %d columns and the maximum is %d.  Functionality will be limited.",
                                inStream.size(), MAX_COLUMNS));
            } else {
                this.largeFileMode = false;
                for (String header : this.headers) {
                    keyColumn.add(header);
                    dataColumns.add(header);
                }
                // Denote we want to include all columns.
                this.lstColumns.getSelectionModel().selectAll();
            }
            // Configure the list-related controls.
            this.lstColumns.setDisable(this.largeFileMode);
            this.allButton.setDisable(this.largeFileMode);
            this.noneButton.setDisable(this.largeFileMode);
            this.flipButton.setDisable(this.largeFileMode);
        }
        // Select the first column as the key.
        this.cmbKeyColumn.getSelectionModel().clearAndSelect(0);
        // Save the file.
        this.inFile = inputFile;
        this.txtInputFile.setText(inputFile.getName());
        // Configure the parent.
        this.parent.configureButtons();
    }

    /**
     * This event fires when the user wants to select a new input file.
     *
     * @param	event	event for button press
     */
    @FXML
    private void selectInput(Event event) {
        File retVal = this.parent.chooseFile(JoinDialog.FLAT_FILTER, "File to Join");
        if (retVal != null) {
            try {
                this.setupFile(retVal);
            } catch (IOException e) {
                BaseController.messageBox(AlertType.ERROR, "Error in Input File", e.toString());
            }
        }
    }

    /**
     * Here the user wants to select all columns.
     *
     * @param	event	event for button press
     */
    @FXML
    private void allColumns(Event event) {
        this.lstColumns.getSelectionModel().selectAll();
    }

    /**
     * Here the user wants to select no columns.
     *
     * @param	event	event for button press
     */
    @FXML
    private void noColumns(Event event) {
        this.lstColumns.getSelectionModel().clearSelection();
    }

    /**
     * Here the user wants to delete the file from the file list.
     *
     * @param event		event for button press
     */
    @FXML
    private void deleteFile(Event event) {
        boolean confirmed = confirmDelete(this.inFile);
        if (confirmed)
            this.parent.deleteFile(this);

    }

    /**
     * Confirm with the user if it's ok to delete a join specification.
     *
     * @param oldFile		current input file
     */
    protected static boolean confirmDelete(File oldFile) {
        // If there is a file in here, get the user to confirm the delete.
        boolean retVal = true;
        if (oldFile != null) {
            Alert a = new Alert(AlertType.CONFIRMATION);
            a.setTitle("Remove Input File");
            a.setContentText("Delete file " + oldFile + " from the input list?");
            Optional<ButtonType> buttonSelected = a.showAndWait();
            retVal = (buttonSelected.isPresent() && buttonSelected.get() == ButtonType.OK);
        }
        return retVal;
    }

    /**
     * Here the user wants to invert the selections.
     *
     * @param event		event for button press
     */
    @FXML
    private void invertColumns(Event event) {
        MultipleSelectionModel<String> selectionModel = this.lstColumns.getSelectionModel();
        // Get a map of the selected items.
        boolean[] selected = new boolean[this.lstColumns.getItems().size()];
        IntStream.range(0, selected.length).forEach(i -> selected[i] = selectionModel.isSelected(i));
        // Clear all current selections.
        selectionModel.clearSelection();
        // Select all the de-selected items from before.
        IntStream.range(0,  selected.length).filter(i -> ! selected[i]).forEach(i -> selectionModel.select(i));
    }

    /**
     * @return TRUE if this join specification is valid, else FALSE
     */
    public boolean isValid() {
        return (this.inFile != null && ! this.cmbKeyColumn.getSelectionModel().isEmpty());
    }

    /**
     * @return the key column name
     */
    public String getKeyColumn() {
        return this.cmbKeyColumn.getSelectionModel().getSelectedItem();
    }

    /**
     * @return the selected list of column headers
     */
    public List<String> getHeaders() {
        // Get the key column name.  The key column is always excluded.
        String keyColumn = this.getKeyColumn();
        // Get the list of column names to use.  This depends on whether or not we are in large-file mode.
        List<String> retVal;
        if (this.largeFileMode)
            retVal = Arrays.stream(this.headers).filter(x -> ! x.contentEquals(keyColumn)).collect(Collectors.toList());
        else
            retVal = this.lstColumns.getSelectionModel().getSelectedItems().stream().filter(x -> ! x.contentEquals(keyColumn)).collect(Collectors.toList());
        return retVal;
    }

    /**
     * This method reads the input file, and creates a linked hash map of the records.  The key is the key column, and the
     * value is a list of data column strings, in the same order as the headers.
     *
     * @return a linked hash map from the key values to the data columns
     *
     * @throws IOException
     */
    public LinkedHashMap<String, List<String>> getRecordMap() throws IOException {
        // Create the hash map.
        int size_estimate = ((int) this.inFile.length()) / 40;
        LinkedHashMap<String, List<String>> retVal = new LinkedHashMap<String, List<String>>(size_estimate);
        // Open the file and get the list of data columns.
        try (TabbedLineReader inStream = new TabbedLineReader(this.inFile)) {
            int[] cols = this.getHeaders().stream().mapToInt(x -> inStream.findColumn(x)).toArray();
            // Verify we found all the fields.
            boolean ok = true;
            for (int i = 0; i < cols.length && ok; i++)
                ok = (cols[i] >= 0);
            if (! ok)
                throw new IOException("Input file " + this.inFile + " has changed on disk.  Reselect and try again.");
            // Get the key column.
            int keyIdx = inStream.findField(this.getKeyColumn());
            // Now read in the file.
            for (TabbedLineReader.Line line : inStream) {
                String key = line.get(keyIdx);
                // Always proceed if ignore-empty is unselected; otherwise, only proceed for
                // non-blank keys.
                if (! this.ignoreEmptyKeys() || ! key.isBlank()) {
                    // We don't use a standard stream here because we need a mutable list.
                    List<String> data = new ArrayList<String>(cols.length);
                    Arrays.stream(cols).forEach(i -> data.add(line.get(i)));
                    // Associate the key with the data.  Note the order is preserved by the linked hash map,
                    // but if there are duplicate keys, the last one will be the one kept.
                    retVal.put(key, data);
                }
            }
        }
        // Return the filled map.
        return retVal;
    }

    /**
     * @return TRUE if we should ignore empty keys, else FALSE
     */
    protected boolean ignoreEmptyKeys() {
        return this.chkIgnoreEmpty.isSelected();
    }

    /**
     * @return the display node for this file specification
     */
    @Override
    public Node getNode() {
        return this.node;
    }

    /**
     * @return the input file
     */
    protected File getInFile() {
        return this.inFile;
    }

    @Override
    public void apply(KeyedFileMap keyedMap) throws IOException {
        // Get the join type.
        Type joinType = this.cmbType.getValue();
        // Process the appropriate join.
        joinType.process(this, keyedMap);
    }

    /**
     * Perform one of the key-based joins.
     *
     * @param keyedMap		incoming file map from the left
     *
     * @throws IOException
     */
    private void processJoin(KeyedFileMap keyedMap) throws IOException {
        // Clear the counters.
        int leftCount = 0;
        int rightDups = 0;
        // Get the join type.
        Type joinType = this.cmbType.getValue();
        // Here we need to read the input file and extract the key column and the data columns.
        File inFile = this.getInFile();
        try (TabbedLineReader inStream = new TabbedLineReader(inFile)) {
            // Create the input map.  This will contain the records in the new file.
            Map<String, List<String>> inputMap = new HashMap<String, List<String>>(keyedMap.size());
            // Get the key column name and index.
            String keyName = this.getKeyColumn();
            int keyIdx = inStream.findField(keyName);
            // Get the names and indices of the data columns.
            List<String> headers = this.getHeaders();
            int[] cols = new int[headers.size()];
            for (int i = 0; i < cols.length; i++)
                cols[i] = inStream.findField(headers.get(i));
            // Add the qualifier to the headers, if needed.
            String qualifier = this.txtQualifier.getText();
            if (! qualifier.isBlank())
                headers = headers.stream().map(x -> qualifier + "." + x).collect(Collectors.toList());
            // Append the headers to the output headers.
            keyedMap.addHeaders(headers);
            // Now read in the file.
            for (TabbedLineReader.Line line : inStream) {
                String key = line.get(keyIdx);
                List<String> data = Arrays.stream(cols).mapToObj(i -> line.get(i)).collect(Collectors.toList());
                if (inputMap.containsKey(key))
                    rightDups++;
                inputMap.put(key, data);
            }
            // Now we iterate through the output map and apply the keyed map.
            Iterator<Map.Entry<String, List<String>>> iter = keyedMap.iterator();
            while (iter.hasNext()) {
                Map.Entry<String, List<String>> current = iter.next();
                String key = current.getKey();
                List<String> data = current.getValue();
                // Is there an input record for this key?
                List<String> newRecord = inputMap.get(key);
                if (newRecord == null) {
                    // No, do special processing.
                    joinType.processMissing(iter, data, headers.size());
                    leftCount++;
                } else {
                    // Yes, add the input columns.
                    data.addAll(newRecord);
                    // Remove the key from the input map.
                    inputMap.remove(key);
                }
            }
            // Update the message display with the stats.
            this.txtMessage.setText(String.format("%d unmatched left keys, %d unmatched right keys, %d duplicates in new file.",
                    leftCount, inputMap.size(), rightDups));
        }
    }

    /**
     * Link two files together using a common internal field rather than the key.  We
     * will organize the input file by the specified key, then locate that field in the
     * incoming output map.  This field, rather than the main key, will be used to find the
     * input-file records matching the left file's records.
     *
     * @param outputMap		incoming file map from the left
     * @throws IOException
     */
    protected void processLink(KeyedFileMap outputMap) throws IOException {
        // Get the key-column name.
        String keyCol = this.getKeyColumn();
        int leftKeyIdx = outputMap.findColumn(keyCol);
        if (leftKeyIdx < 0)
            throw new IOException("Cannot find column \"" + keyCol + "\" on the left.");
        // Get the input file (on the right), keyed by the key column.
        KeyedFileMap inputMap = new KeyedFileMap(this.inFile, keyCol);
        // Update the headers for the output map.
        List<String> headers = this.getHeaders();
        outputMap.addHeaders(headers);
        // Get the locations of the input file data columns.
        int[] colIdxs = headers.stream().mapToInt(x -> inputMap.findColumn(x)).toArray();
        // These will be used for our output statistics.
        int dupCount = 0;
        int leftOnly = 0;
        Set<String> keysUsed = new HashSet<String>(inputMap.size());
        // Loop through the output map, linking records.  Note that we do not have
        // unique keys any more, so the same input map record may be used multiple
        // times.
        Iterator<Map.Entry<String, List<String>>> iter = outputMap.iterator();
        while (iter.hasNext()) {
            Map.Entry<String, List<String>> outEntry = iter.next();
            List<String> outRecord = outEntry.getValue();
            String key = outRecord.get(leftKeyIdx);
            List<String> inRecord = inputMap.getRecord(key);
            if (inRecord == null) {
                // No matching right record, so delete this left record.
                leftOnly++;
                iter.remove();
            } else {
                // Here we are keeping the record. Update the duplicate-key count.
                if (! keysUsed.add(key))
                    dupCount++;
                // Add the new-record columns to the output record.
                Arrays.stream(colIdxs).forEach(i -> outRecord.add(inRecord.get(i)));
            }
        }
        // Update the status message.
        this.txtMessage.setText(String.format("%d keys missing on the left, %d keys unused on the right, %d keys reused, %d output records.",
                leftOnly, inputMap.size() - keysUsed.size(), dupCount, outputMap.size()));
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
