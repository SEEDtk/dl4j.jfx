/**
 *
 */
package org.theseed.join;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.io.KeyedFileMap;
import org.theseed.io.TabbedLineReader;
import org.theseed.jfx.BaseController;

import javafx.collections.ObservableList;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.Alert.AlertType;

/**
 * This is the base class for the two filtering operations.
 *
 * @author Bruce Parrello
 *
 */
public class FilterSpec implements IJoinSpec {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(JoinSpec.class);
    /** input file */
    private File inFile;
    /** parent dialog */
    private JoinDialog parent;
    /** display node for this join specification */
    private Node node;

    // CONTROLS

    /** input file name */
    @FXML
    private TextField txtInputFile;

    /** key column selector */
    @FXML
    private ChoiceBox<String> cmbKeyColumn;

    /** specification title */
    @FXML
    private Label lblTitle;

    /** type of filtering */
    @FXML
    private ChoiceBox<Type> cmbType;

    /** stats message area */
    @FXML
    private TextArea txtMessage;

    public static enum Type {
        INCLUDE {
            @Override
            public String toString() {
                return "include";
            }

            @Override
            public void apply(FilterSpec parent, KeyedFileMap outputMap) throws IOException {
                parent.applyFilter(outputMap, true);

            }

        }, EXCLUDE {
            @Override
            public String toString() {
                return "exclude";
            }

            @Override
            public void apply(FilterSpec parent, KeyedFileMap outputMap) throws IOException {
                parent.applyFilter(outputMap, false);
            }

        }, MERGE {
            @Override
            public String toString() {
                return "merge";
            }

            @Override
            public void apply(FilterSpec parent, KeyedFileMap outputMap) throws IOException {
                parent.applyMerge(outputMap);
            }

        }, FLAG {
            @Override
            public String toString() {
                return "flag";
            }

            @Override
            public void apply(FilterSpec parent, KeyedFileMap outputMap) throws IOException {
                parent.applyFlag(outputMap);
            }

        };

        /**
         * @return the string representation of the type
         */
        public abstract String toString();

        /**
         * Apply this filter to the output map.
         *
         * @param parent		parent filter-spec object
         * @param outputMap		output map to filter
         *
         * @throws IOException
         */
        public abstract void apply(FilterSpec parent, KeyedFileMap outputMap) throws IOException;

    }

    @Override
    public void init(JoinDialog parent, Node node) {
        // Save the parent dialog information.
        this.parent = parent;
        this.node = node;
        // Denote there is no input file.
        this.inFile = null;
        // Set up the type-selection combo box.
        this.cmbType.getItems().addAll(Type.values());
        this.cmbType.getSelectionModel().clearAndSelect(0);
    }

    @Override
    public boolean isValid() {
        return (this.inFile != null && ! this.cmbKeyColumn.getSelectionModel().isEmpty());
    }

    @Override
    public Node getNode() {
        return this.node;
    }

    /**
     * Here the user has selected a file as the input file.  We update the
     * file name display and fill in the list box and the choice box.
     *
     * @param inputFile		new input file to use
     *
     * @throws IOException
     */
    private void setupFile(File inputFile) throws IOException {
        // Remember this directory.
        this.parent.setParentDirectory(inputFile.getParentFile());
        // Get the column headers for this file and fill the controls.
        ObservableList<String> keyColumn = this.cmbKeyColumn.getItems();
        try (TabbedLineReader inStream = new TabbedLineReader(inputFile)) {
            if (inStream.size() < 1)
                throw new IOException("Input file must contain at least one column.");
            for (String header : inStream.getLabels())
                keyColumn.add(header);
        }
        // Select the first column as the key.
        this.cmbKeyColumn.getSelectionModel().clearAndSelect(0);
        // Default to natural join.
        this.cmbType.getSelectionModel().select(Type.INCLUDE);
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
        File retVal = this.parent.chooseFile(JoinDialog.FLAT_FILTER, "Filtering File");
        if (retVal != null) {
            try {
                this.setupFile(retVal);
            } catch (IOException e) {
                BaseController.messageBox(AlertType.ERROR, "Error in Input File", e.toString());
            }
        }
    }

    /**
     * Here the user wants to delete the file from the file list.
     *
     * @param event		event for button press
     */
    @FXML
    private void deleteFile(Event event) {
        boolean confirmed = JoinSpec.confirmDelete(this.inFile);
        if (confirmed)
            this.parent.deleteFile(this);
    }

    @Override
    public void apply(KeyedFileMap outputMap) throws IOException {
        // Get the filtering type.
        Type filterType = this.cmbType.getValue();
        // Apply the filter.
        filterType.apply(this, outputMap);
    }

    /**
     * Merge the files.  The columns are reduced to the columns both files have
     * in common.  The input records are then added to the output map.  If a
     * key already exists in the output map, its record will be replaced with
     * the new data.
     *
     * @param outputMap		output map being build
     *
     * @throws IOException
     */
    private void applyMerge(KeyedFileMap outputMap) throws IOException {
        // Get the input file.
        try (TabbedLineReader inStream = new TabbedLineReader(this.inFile)) {
            // Get all the data columns in the input file.
            String keyCol = this.cmbKeyColumn.getValue();
            Set<String> newHeaders = Arrays.stream(inStream.getLabels())
                    .filter(x -> ! x.contentEquals(keyCol)).collect(Collectors.toSet());
            // Save the current width and duplicate-key count.  We'll use these
            // in our statistics report.
            int oldWidth = outputMap.width();
            int oldDups = outputMap.getDupCount();
            // Count the number of records added.
            int newRecords = 0;
            // Remove other columns from the output map.
            outputMap.reduceCols(newHeaders);
            // Get the new list of headers.
            List<String> outHeaders = outputMap.getHeaders();
            // Get the index of the key column in the input file.
            int keyIdx = inStream.findColumn(keyCol);
            // Compute the indices of the data columns.
            int[] dataCols = IntStream.range(1, outHeaders.size())
                    .map(i -> inStream.findColumn(outHeaders.get(i)))
                    .toArray();
            // Now we loop through the input file, adding records.
            for (TabbedLineReader.Line line : inStream) {
                String key = line.get(keyIdx);
                List<String> data = Arrays.stream(dataCols).mapToObj(i -> line.get(i))
                        .collect(Collectors.toList());
                newRecords++;
                outputMap.addRecord(key, data);
            }
            // Output the statistics.
            int replacements = outputMap.getDupCount() - oldDups;
            this.txtMessage.setText(String.format("%d columns removed, %d records added, %d replaced.",
                    oldWidth - outputMap.width(), newRecords - replacements, replacements));
        }
    }

    /**
     * Apply normal filtering.  In this case, the keys in the output map are
     * compared with the keys in the input file and the output map records are
     * either kept or discarded.
     *
     * @param outputMap		output map being build
     * @param type			TRUE to keep only matching records, FALSE to remove them instead
     *
     * @throws IOException
     */
    private void applyFilter(KeyedFileMap outputMap, boolean type) throws IOException {
        // Get the set of keys in this input file.
        Set<String> keys = getInputKeys();
        // Remember the initial file size.
        int initial = outputMap.size();
        // Loop through the output map, checking each record against the set.
        Iterator<Map.Entry<String, List<String>>> iter = outputMap.iterator();
        while (iter.hasNext()) {
            String key = iter.next().getKey();
            if (keys.contains(key) != type)
                iter.remove();
        }
        // Write the status.
        this.txtMessage.setText(String.format("%d keys in filter file.  %d records kept, %d deleted.",
                keys.size(), outputMap.size(), initial - outputMap.size()));
    }

    /**
     * @return the set of keys in the input file
     *
     * @throws IOException
     */
    protected Set<String> getInputKeys() throws IOException {
        String keyColumn = this.cmbKeyColumn.getValue();
        Set<String> retVal = TabbedLineReader.readSet(this.inFile, keyColumn);
        return retVal;
    }

    /**
     * Add a flag column at the end.  The column will contain "Y" if there is a match, and will be blank
     * otherwise.
     *
     * @param outputMap		output map being built
     */
    protected void applyFlag(KeyedFileMap outputMap) throws IOException {
        // Get the base name of the input file without the extension.  This is our column title.
        String name = StringUtils.substringBeforeLast(inFile.getName(), ".");
        outputMap.addHeaders(Arrays.asList(name));
        // Get all the keys in the input file.
        Set<String> keys = this.getInputKeys();
        // Loop through the output, adding the flag value.
        int count = 0;
        for (List<String> record : outputMap.getRecords()) {
            // The first field is always the key.
            String flag = (keys.contains(record.get(0)) ? "Y" : "");
            record.add(flag);
            count++;
        }
        this.txtMessage.setText(String.format("%d records flagged.", count));
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
