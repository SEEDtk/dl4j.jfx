/**
 *
 */
package org.theseed.join;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.io.KeyedFileMap;
import org.theseed.io.LineReader;
import org.theseed.jfx.BaseController;

import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.Alert.AlertType;
import javafx.stage.FileChooser.ExtensionFilter;

/**
 * This specification performs a classification analysis, comparing a column of predicted classes
 * with a column of actual classes.  We generate the confusion matrix with both raw numbers and
 * percentages, and then output the usual metrics.  This is an output method, but it does not
 * change the keyed-map flowing through.  What is output is the very non-standard format
 * confusion report.
 *
 * @author Bruce Parrello
 *
 */
public class ConfusionSpec extends SaveSpec {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(ConfusionSpec.class);
    /** label file name */
    private File labelFile;
    /** ordered list of labels */
    private List<String> labels;
    /** column width in confusion matrix */
    private static final int COL_WIDTH = 10;
    /** integer format for confusion matrix */
    private static final String INTEGER_FORMAT = "%" + Integer.toString(COL_WIDTH) + "d";
    /** float format for confusion matrix */
    private static final String PERCENT_FORMAT = "%" + Integer.toString(COL_WIDTH) + ".2f";
    /** string format for confusion matrix */
    private static final String STRING_FORMAT = "%-" + Integer.toString(COL_WIDTH) + "s";
    /** string format for confusion matrix headers */
    private static final String HEADER_FORMAT = "%" + Integer.toString(COL_WIDTH) + "s ";
    /** statistics format */
    private static final String STATS_FORMAT = "%-20s %" + Integer.toString(COL_WIDTH) + ".4f%n";

    /**
     * This is a handler to update the parent dialog button states when one
     * of the column names changes.
     */
    private class ChangeHandler implements ChangeListener<String> {

        @Override
        public void changed(ObservableValue<? extends String> observable, String oldValue, String newValue) {
            ConfusionSpec.this.getParent().configureButtons();
        }

    }

    // CONTROLS

    /** name of expected-class column */
    @FXML
    private TextField txtExpectColumn;

    /** name of predicted-class column */
    @FXML
    private TextField txtPredictColumn;

    /** input label file name */
    @FXML
    private TextField txtLabelFile;

    /** output message area */
    @FXML
    private TextArea txtMessage;

    @Override
    public boolean isValid() {
        return ! StringUtils.isAnyBlank(this.txtExpectColumn.getText(), this.txtPredictColumn.getText())
                && this.labelFile != null && this.getOutFile() != null;
    }

    @Override
    public void apply(KeyedFileMap keyedMap) throws IOException {
        // Find the expected column and the predicted column.
        int expectColIdx = keyedMap.findColumn(this.txtExpectColumn.getText());
        if (expectColIdx < 0)
            throw new IOException("Could not find Expect Column.");
        int predictColIdx = keyedMap.findColumn(this.txtPredictColumn.getText());
        if (predictColIdx < 0)
            throw new IOException("Could not find Predict Column.");
        // Read the labels from the label file.
        this.labels = LineReader.readList(this.labelFile);
        // Now we must build the confusion matrix.  The first matrix dimension
        // is the expected class, and the second is the predicted class.
        int[][] matrix = this.labels.stream()
                .map(x -> new int[labels.size()]).toArray(int[][]::new);
        // These will count the total number of rows, the total number
        // of correct predictions, and the total number of invalid rows.
        int recordCount = 0;
        int correctCount = 0;
        int errorCount = 0;
        // Loop through the records.
        for (List<String> record : keyedMap.getRecords()) {
            // Get the array indices for the expected and predicted values.
            int expectIdx = this.idxLabel(record.get(expectColIdx));
            int predictIdx = this.idxLabel(record.get(predictColIdx));
            if (expectIdx < 0 || predictIdx < 0)
                errorCount++;
            else {
                // Here we have a valid record.
                recordCount++;
                if (expectIdx == predictIdx)
                    correctCount++;
                // Update the appropriate matrix entry.
                matrix[expectIdx][predictIdx]++;
            }
        }
        // Error out of there was simply no data.  This means the column names are probably wrong.
        if (recordCount == 0)
            throw new IOException("No valid prediction records found.");
        // Here we abbreviate all of the labels so that they will fit in our standard format.
        String[] abbrs = this.labels.stream().map(x -> StringUtils.abbreviate(x, COL_WIDTH))
                .toArray(String[]::new);
        // Now we have all our data.  It is time to make the report.
        try (PrintWriter writer = new PrintWriter(this.getOutFile())) {
            // Start with the most important raw numbers.
            writer.format("%d predictions found.  %d were invalid.%n", recordCount, errorCount);
            writer.format("%d correct predictions (accuracy %1.2f%%).%n", correctCount, (correctCount * 100.0 / recordCount));
            // Write the raw confusion matrix.
            writer.println();
            writer.println("Confusion matrices compare the expected (row) value to predicted (column)");
            // Start with the header row.
            this.writeConfusionHeader(abbrs, writer, "Raw");
            // Loop through the data rows.
            for (int r = 0; r < abbrs.length; r++) {
                writer.format(STRING_FORMAT + "|", abbrs[r]);
                for (int c = 0; c < abbrs.length; c++)
                    writer.format(INTEGER_FORMAT + " ", matrix[r][c]);
                writer.println();
            }
            // Write the percentage confusion matrix.
            this.writeConfusionHeader(abbrs, writer, "Percent");
            for (int r = 0; r < abbrs.length; r++) {
                writer.format(STRING_FORMAT + "|", abbrs[r]);
                for (int c = 0; c < abbrs.length; c++)
                    writer.format(PERCENT_FORMAT + " ", (matrix[r][c] * 100.0) / recordCount);
                writer.println();
            }
            // Explain the stats.
            writer.println();
            writer.println("Accuracy is how often the class in question is predicted correctly.");
            writer.println("  Accuracy = (correctly predicted) / (all records)");
            writer.println("Precision is how often objects belonging to the class were found.");
            writer.println("  Precision = (true positive) / (all positive)");
            writer.println("Sensitivity is how likely an object found belongs to the class.");
            writer.println("  Sensitivity = (true positive) / (predicted positive)");
            writer.println("Fallout is how likely a wrong object will be found.");
            writer.println("  Fallout = (false positive) / (all negative)");
            // Do the statistics for each class.
            for (int i = 0; i < abbrs.length; i++)
                this.printMetrics(writer, matrix, i);
        }
        // Finally, open the output file.
        this.checkForOpen();
    }


    /**
     * Write out the metrics for a single classification label.  The label is treated
     * as the positive class, and all other labels as the negative class.
     *
     * @param writer	output writer
     * @param matrix	confusion matrix
     * @param idx		index of the classification label of interest
     */
    private void printMetrics(PrintWriter writer, int[][] matrix, int idx) {
        writer.println();
        writer.println();
        writer.format("Classification metrics for %s%n", this.labels.get(idx));
        writer.println();
        // This is the number of classes.
        int n = this.labels.size();
        // Compute the true positive, true negative, false positive, and false negative.
        int tp = 0;
        int fp = 0;
        int fn = 0;
        int tn = 0;
        for (int r = 0; r < n; r++)
            for (int c = 0; c < n; c++) {
                int val = matrix[r][c];
                if (c == idx) {
                    if (r == idx)
                        tp += val;
                    else
                        fp += val;
                } else {
                    if (r == idx)
                        fn += val;
                    else
                        tn += val;
                }
            }
        writer.format(STATS_FORMAT, "Class Accuracy", safeRatio(tp+tn, tp+fp+tn+fn));
        writer.format(STATS_FORMAT, "Class Precision", safeRatio(tp, tp+fp));
        writer.format(STATS_FORMAT, "Class Sensitivity", safeRatio(tp, tp+fn));
        writer.format(STATS_FORMAT, "Class Fallout", safeRatio(fp, fp+tn));
        writer.format(STATS_FORMAT, "False Discovery", safeRatio(fp, fp+tp));
    }

    /**
     * Write the header row of a confusion matrix.
     *
     * @param abbrs		label abbreviations
     * @param writer	output writer
     * @param type		type of matrix
     */
    private void writeConfusionHeader(String[] abbrs, PrintWriter writer, String type) {
        writer.println();
        writer.format(STRING_FORMAT + "|", type);
        Arrays.stream(abbrs).forEach(x -> writer.format(HEADER_FORMAT, x));
        writer.println();
        String spacer = StringUtils.repeat('-', (COL_WIDTH+1)*abbrs.length + COL_WIDTH);
        writer.println(spacer);
    }

    /**
     * Compute a ratio safely.
     *
     * @param num		number of items of interest
     * @param denom		total number of items
     *
     * @return the fraction of items that were of interest
     */
    private static double safeRatio(int num, int denom) {
        double retVal = 0.0;
        if (denom > 0)
            retVal = ((double) num) / denom;
        return retVal;
    }

    @Override
    protected void initControls() {
        // Set up the change handlers for the column names.
        ChangeListener<String> handler = this.new ChangeHandler();
        this.txtExpectColumn.textProperty().addListener(handler);
        this.txtPredictColumn.textProperty().addListener(handler);
    }

    @Override
    protected String getFileLabel() {
        return "Confusion Report";
    }

    @Override
    protected ExtensionFilter getFilter() {
        return JoinDialog.FLAT_FILTER;
    }

    /**
     * @return the array index of the specified label, or -1 if it is invalid
     *
     * @param label		label to check
     */
    private int idxLabel(String label) {
        int retVal = this.labels.size() - 1;
        while (retVal > 0 && ! this.labels.get(retVal).contentEquals(label))
            retVal--;
        return retVal;
    }

    /**
     * Here the user wants to select a new label file.
     *
     * @param event		event for the button press
     */
    @FXML
    private void selectLabels(Event event) {
        try {
            File newFile = this.getParent().chooseFile(AnalyzeSpec.LABEL_FILTER, "Label File");
            if (newFile != null) {
                // Set up the new file.
                this.labelFile = newFile;
                this.txtLabelFile.setText(newFile.getName());
                // Configure the join dialog buttons.
                this.getParent().configureButtons();
            }
        } catch (Exception e) {
            BaseController.messageBox(AlertType.ERROR, "Error in Label File", e.toString());
        }
    }

}
