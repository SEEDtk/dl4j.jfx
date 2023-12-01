/**
 *
 */
package org.theseed.join;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.util.DoubleArray;
import org.apache.commons.math3.util.ResizableDoubleArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.io.KeyedFileMap;
import org.theseed.io.LineReader;
import org.theseed.jfx.BaseController;
import javafx.collections.ObservableList;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.stage.FileChooser.ExtensionFilter;

/**
 * This specification requests a column analysis.  The theory is that the incoming file represents
 * the training/testing file for a data model.  The key column will be ignored.  All metadata
 * columns should have been removed in previous steps.  In general, non-numeric data will be
 * skipped, and columns with no numeric data will be discarded.  The output will have a new
 * key column consisting of the input column titles.  There will be one data column for
 * each incoming label.  If the file appears to be a classification set (one column consists
 * entirely of labels), then a mean bias will be performed.  If the file appears to be a regression
 * set (each label is represented by a column), then a pearson-coefficient analysis will be
 * performed.  If there is more than one label, the last data column will contain the identity
 * of the label whose analysis value has the highest absolute value (indicating the strongest
 * correlation).
 *
 * @author Bruce Parrello
 *
 */
public class AnalyzeSpec implements IJoinSpec {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(AnalyzeSpec.class);
    /** parent dialog */
    private JoinDialog parent;
    /** display node for this join specification */
    private Node node;
    /** current label file */
    private File labelFile;
    /** list of labels */
    private List<String> labels;
    /** array of label counts */
    private LabelCount[] labelCounters;
    /** extension filter for label files */
    protected static final ExtensionFilter LABEL_FILTER = new ExtensionFilter("Label files", "*.txt");

    // CONTROLS

    /** specification title */
    @FXML
    private Label lblTitle;

    /** input label file name */
    @FXML
    private TextField txtLabelFile;

    /** table of label names and counts */
    @FXML
    private TableView<LabelCount> tblLabels;

    /** label name column */
    @FXML
    private TableColumn<LabelCount, String> colLabel;

    /** label count column */
    @FXML
    private TableColumn<LabelCount, Integer> colCount;

    /** column name for new first column */
    @FXML
    private TextField txtMetaColName;

    /**
     * This class is used to represent a label in the label table.  After processing,
     * it will display the label counts.
     */
    public class LabelCount {

        private String name;
        private int count;

        /**
         * Create a label count.
         */
        public LabelCount(String labelName) {
            this.name = labelName;
            this.count = 0;
        }

        /**
         * Increment the count.
         */
        public void count() {
            this.count++;
        }

        /**
         * @return the label name
         */
        public String getName() {
            return this.name;
        }

        /**
         * @return the label count
         */
        public Integer getCount() {
            return this.count;
        }

        /**
         * Set a new name.
         *
         * @param name		new name to set
         */
        public void setName(String name) {
            this.name = name;
        }

        /**
         * Set a new count.
         *
         * @param count 	new count to set
         */
        public void setCount(Integer count) {
            this.count = count;
        }

    }



    @Override
    public void init(JoinDialog parent, Node node) {
        // Save the parent and the node.
        this.parent = parent;
        this.node = node;
        // Initialize the table column display factories.
        this.colLabel.setCellValueFactory(new PropertyValueFactory<LabelCount, String>("name"));
        this.colCount.setCellValueFactory(new PropertyValueFactory<LabelCount, Integer>("count"));
        // Check for a convenient label file.
        File testFile = new File(parent.getParentDirectory(), "labels.txt");
        boolean ok = false;
        if (testFile.exists()) {
            // We have a local label file, so set it up.
            try {
                this.setupFile(testFile);
                ok = true;
            } catch (Exception e) {
                // If an error occurred, fail silently.  The file was apparently bogus.
            }
        }
        if (! ok) {
            // No local label file, so set us up as empty.
            this.labelFile = null;
            this.labels = Collections.emptyList();
            this.labelCounters = null;
        }
    }

    /**
     * Analyze a new label file and get the label list.
     *
     * @param newFile	new label file
     *
     * @throws IOException
     */
    private void setupFile(File newFile) throws IOException {
        // Read the label list from the file.
        this.labels = LineReader.readList(newFile);
        // If we got this far, the file has actual labels in it.  Save them in the table
        // control.
        ObservableList<LabelCount> target = this.tblLabels.getItems();
        target.clear();
        this.labelCounters = this.labels.stream().map(x -> new LabelCount(x)).toArray(LabelCount[]::new);
        target.addAll(this.labelCounters);
        // Save the file name.
        this.labelFile = newFile;
        this.txtLabelFile.setText(newFile.getName());
    }

    @Override
    public boolean isValid() {
        return (this.labels.size() >= 1 && ! this.txtMetaColName.getText().isBlank());
    }

    @Override
    public void apply(KeyedFileMap keyedMap) throws IOException {
        // This will be the new file map.
        KeyedFileMap newMap;
        // Get the map headers.
        List<String> headers = keyedMap.getHeaders();
        // Convert the incoming file to floating-point.
        List<double[]> numRecords = keyedMap.getRecordNumbers();
        // We must now determine the type of this operation.  If each label is
        // also a column header, this is a regression file; otherwise, it is
        // classification.
        if (headers.containsAll(this.labels)) {
            // Here we have a regression file.  Isolate the label columns.
            int[] labelCols = this.findLabelColumns(headers);
            // Compute the pearson coefficients.
            newMap = this.processRegression(headers, numRecords, labelCols);
        } else {
            // Here we have a classification file.  Determine the output column.
            int labelColIdx = this.findLabelColumn(keyedMap);
            // Compute the label index for each row.
            int[] rowLabels = keyedMap.getRecords().stream().mapToInt(x -> this.labels.indexOf(x.get(labelColIdx)))
                    .toArray();
            // Do the classification analysis.
            newMap = this.processClassification(headers, numRecords, rowLabels, labelColIdx);
        }
        // Add the mean value of each column at the end.
        this.computeMeans(headers, numRecords, newMap);
        // Update the counts in the table view.
        this.tblLabels.refresh();
        // We must clone the new keyed map into the old one.
        keyedMap.shallowCopyFrom(newMap);
    }


    /**
     * This method computes the mean value of each input file column and adds it to the output file
     * as an additional column.  For a 1/0 input column, this will tell us fraction of times the
     * column value is 1.  For others, it gives us an idea of the baseline value.
     *
     * @param header		header list containing the column names
     * @param numRecords	list of input file records encoded as floating-point arrays
     * @param newMap		output file map
     */
    private void computeMeans(List<String> headers, List<double[]> numRecords, KeyedFileMap newMap) {
        newMap.addHeaders(Arrays.asList("col_mean"));
        // We go through the header list and the keyed file map records in parallel. The file map
        // records are a subset of the header elements in the same order.  Our position in the
        // header list gives us the appropriate column index.  When we find a match, we compute
        // the mean of the column and add it at the end of the output record.
        int c = 1;
        for (List<String> record : newMap.getRecords()) {
            // Get the key of this record.
            String key = record.get(0);
            // Find the correct position in the headers.  If we fail, it will cause an index out-of-bounds
            // exception, which we reframe to give the user more useful information.  It should, however,
            // never happen.
            try {
                while (! headers.get(c).contentEquals(key)) c++;
            } catch (IndexOutOfBoundsException e) {
                throw new IllegalStateException("Column name \"" + key + "\" not found in headers or out of order.");
            }
            // Now "c" is the column index for this output record.  Compute the mean.
            double total = 0.0;
            int found = 0;
            for (double[] numRecord : numRecords) {
                if (Double.isFinite(numRecord[c])) {
                    total += numRecord[c];
                    found++;
                }
            }
            if (found > 0)
                total /= found;
            // Add the mean to the output record.
            record.add(Double.toString(total));
        }

    }

    /**
     * Compute the mean bias of each label for each data column in the incoming map.
     *
     * @param headers		list of column headers
     * @param numRecords	data records in numeric form
     * @param rowLabels 	array of label indices for each data row
     * @param labelColIdx	index of the label column
     *
     * @return a new keyed file map containing the column names as keys and the mean values per label as data
     *
     */
    private KeyedFileMap processClassification(List<String> headers, List<double[]> numRecords,
            int[] rowLabels, int labelColIdx) {
        // This will be the output map.
        KeyedFileMap retVal = this.buildNewMap();
        // Count the row labels.
        for (int labelIdx : rowLabels)
            this.labelCounters[labelIdx].count();
        // This is a holding area, initially for the totals but ultimately for the results.
        Map<String, double[]> statsMap = createStatsMap(numRecords);
        // Loop through the data columns, building totals for each label.
        for (int c = 1; c < headers.size(); c++) {
            // Note we skip the label column.
            if (c != labelColIdx) {
                // These will hold the results for each label.
                double[] results = new double[this.labels.size()];
                // Now loop through the records and the row labels in parallel.
                int  found = 0;
                for (int r = 0; r < numRecords.size(); r++) {
                    double val = numRecords.get(r)[c];
                    if (Double.isFinite(val)) {
                        results[rowLabels[r]] += val;
                        found++;
                    }
                }
                // If this column had any data in it, store the statistics in the map.
                if (found > 0) {
                    // Turn the totals into means.
                    for (int i = 0; i < results.length; i++)
                        results[i] /= (double) found;
                    // Store them in the stats map.
                    statsMap.put(headers.get(c), results);
                }
            }
        }
        // Prepare the output map.  We need an extra column for the best label.
        retVal.addHeaders(Arrays.asList("best"));
        for (Map.Entry<String, double[]> statsEntry : statsMap.entrySet()) {
            String colName = statsEntry.getKey();
            double[] means = statsEntry.getValue();
            // Compute the best value here.
            int maxI = 0;
            double max = means[0];
            for (int i = 1; i < means.length; i++) {
                if (means[i] > max) {
                    maxI = i;
                    max = means[i];
                }
            }
            String best = this.labels.get(maxI);
            // Now form the output record.
            List<String> record = new ArrayList<String>(means.length + 1);
            for (double mean : means)
                record.add(Double.toString(mean));
            record.add(best);
            retVal.addRecord(colName, record);
        }
        // Return the new output file map.
        return retVal;
    }

    /**
     * @return an empty statistical map for storing results
     *
     * @param numRecords	incoming data records
     */
    private Map<String, double[]> createStatsMap(List<double[]> numRecords) {
        return new LinkedHashMap<String, double[]>(numRecords.size() * 3 / 2);
    }

    /**
     * @return a new map to hold the data on each column/label combination
     */
    private KeyedFileMap buildNewMap() {
        KeyedFileMap retVal = new KeyedFileMap(txtMetaColName.getText());
        retVal.addHeaders(this.labels);
        return retVal;
    }

    /**
     * Here we have a classification file, and we want to find the column containing the
     * labels.
     *
     * @param keyedMap		current file map
     *
     * @return the index of the label column in the current file
     *
     * @throws IOException
     */
    private int findLabelColumn(KeyedFileMap keyedMap) throws IOException {
        // Get the full set of records.
        Collection<List<String>> records = keyedMap.getRecords();
        // Get a set of the labels to speed up the checks.
        Set<String> labelSet = this.labels.stream().collect(Collectors.toSet());
        // Generally the label column is at the end, so we search backward.
        int width = keyedMap.getHeaders().size();
        int retVal = -1;
        for (int c = width - 1; c > 0 && retVal < 0; c--) {
            // This will be set to FALSE if we find a non-label.
            boolean found = true;
            Iterator<List<String>> iter = records.iterator();
            while (iter.hasNext() && found) {
                String colData = iter.next().get(c);
                found = labelSet.contains(colData);
            }
            if (found)
                retVal = c;
        }
        if (retVal < 0) {
            // Here there is no label column.
            throw new IOException("Label column not found in classification file.");
        }
        return retVal;
    }

    /**
     * Compute a Pearson Correlation between each data column and every label column.
     * A record will be constructed containing as the key the column header and as
     * the data the pearson coefficients in order.
     *
     * @param headers		list of column headers
     * @param numRecords	data records in numeric form
     * @param labelCols		array of label column indices
     *
     * @return a new keyed file map containing the column names as keys and the pearson coefficients per label as data
     */
    private KeyedFileMap processRegression(List<String> headers, List<double[]> numRecords, int[] labelCols) {
        // We need this to compute the correlations.
        PearsonsCorrelation computer = new PearsonsCorrelation();
        // This will be the output map.
        KeyedFileMap retVal = this.buildNewMap();
        // This will track the number of output columns.
        int width = labelCols.length;
        // Add the "best" column if there are multiple labels.
        if (labelCols.length > 1) {
            retVal.addHeaders(Arrays.asList("best"));
            width++;
        }
        // Count the number of good values in each label column.  In general, we don't expect any
        // bad values.  If they show up, it is a huge red flag.
        for (double[] record : numRecords) {
            for (int lbl = 0; lbl < labelCols.length; lbl++) {
                if (Double.isFinite(record[labelCols[lbl]]))
                    this.labelCounters[lbl].count();
            }
        }
        // These sub-arrays are used to hold the column data.  When we extract a pair of columns, we remove
        // every row where one of the values is non-finite.
        DoubleArray colValues = new ResizableDoubleArray(numRecords.size());
        DoubleArray labelColValues = new ResizableDoubleArray(numRecords.size());
        // Now we loop through all of the columns.  We skip the key column and the label columns.
        // For each column, we compute the pearson coefficient between that column and every label
        // column.
        for (int c = 1; c < headers.size(); c++) {
            if (! ArrayUtils.contains(labelCols, c)) {
                double[] results = new double[labelCols.length];
                // This will remember the best correlation found.
                double best = -1.0;
                String bestLabel = "";
                // Loop through the labels.
                for (int lbl = 0; lbl < labelCols.length; lbl++) {
                    int l = labelCols[lbl];
                    // Build the parallel arrays.
                    colValues.clear();
                    labelColValues.clear();
                    for (int r = 0; r < numRecords.size(); r++) {
                        double[] record = numRecords.get(r);
                        double val = record[c];
                        double labelVal = record[l];
                        if (Double.isFinite(val) && Double.isFinite(labelVal)) {
                            colValues.addElement(val);
                            labelColValues.addElement(labelVal);
                        }
                    }
                    // We can only proceed if we have at least two rows.
                    if (colValues.getNumElements() < 2)
                        results[lbl] = Double.NaN;
                    else
                        results[lbl] = computer.correlation(colValues.getElements(), labelColValues.getElements());
                    if (Double.isFinite(results[lbl])) {
                        // Here we have a good results.  Update the best-label information.  For a pearson
                        // correlation, we compare absolute values instead of real values.
                        double absValue = Math.abs(results[lbl]);
                        if (absValue > best) {
                            bestLabel = this.labels.get(lbl);
                            best = absValue;
                        }
                    }
                }
                // If we got at least one value, output a row.
                if (! bestLabel.isEmpty()) {
                    List<String> record = new ArrayList<String>(width);
                    for (double result : results)
                        record.add(Double.toString(result));
                    if (width > record.size())
                        record.add(bestLabel);
                    retVal.addRecord(headers.get(c), record);
                }
            }
        }
        // Return the new output map.
        return retVal;
    }

    /**
     * Locate all of the label columns in a regression input file.  This method will
     * not be called unless the label columns are known to exist.
     *
     * @param headers	list of file headers
     *
     * @return an array of the label column indices
     */
    private int[] findLabelColumns(List<String> headers) {
        // Allocate the output array.
        int[] retVal = this.labels.stream().mapToInt(x -> headers.lastIndexOf(x)).toArray();
        // Return it to the caller.
        return retVal;
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
     * Here the user wants to select a new label file.
     *
     * @param event		event for the button press
     */
    @FXML
    private void selectLabels(Event event) {
        try {
            File newFile = this.parent.chooseFile(LABEL_FILTER, "Label File");
            if (newFile != null) {
                // Set up the new file.
                this.setupFile(newFile);
                // Configure the join dialog buttons.
                this.parent.configureButtons();
            }
        } catch (Exception e) {
            BaseController.messageBox(AlertType.ERROR, "Error in Label File", e.toString());
        }
    }

    /**
     * Here the user wants to delete the file spec from the file list.
     *
     * @param event		event for button press
     */
    @FXML
    private void deleteFile(Event event) {
        boolean confirmed = JoinSpec.confirmDelete(this.labelFile);
        if (confirmed)
            this.parent.deleteFile(this);
    }

}
