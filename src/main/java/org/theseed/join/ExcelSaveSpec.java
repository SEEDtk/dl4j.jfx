/**
 *
 */
package org.theseed.join;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.excel.CustomWorkbook;
import org.theseed.io.KeyedFileMap;

import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.stage.FileChooser.ExtensionFilter;

/**
 * This saves a current copy of the output in an excel spreadsheet as a table.
 *
 * @author Bruce Parrello
 *
 */
public class ExcelSaveSpec extends SaveSpec {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(ExcelSaveSpec.class);
    /** number format precision */
    private int precision;
    /** format for creating a pubmed link */
    protected static final String PUBMED_FORMAT = "https://pubmed.ncbi.nlm.nih.gov/%s";
    /** width factor for spreadsheets */
    private static final int WIDTH_FACTOR = 300;

    public static enum MaxWidth {
        M25 {
            @Override
            public int getWidth() {
                return 25 * WIDTH_FACTOR;
            }

            @Override
            public String toString() {
                return "25";
            }
        }, M50 {
            @Override
            public int getWidth() {
                return 50 * WIDTH_FACTOR;
            }

            @Override
            public String toString() {
                return "50";
            }
        }, M100 {
            @Override
            public int getWidth() {
                return 100 * WIDTH_FACTOR;
            }

            @Override
            public String toString() {
                return "100";
            }
        }, UNLIMITED {
            @Override
            public int getWidth() {
                return Integer.MAX_VALUE;
            }

            @Override
            public String toString() {
                return "unlimited";
            }
        };

        /**
         * @return the maximum width allowed
         */
        public abstract int getWidth();

    }

    // CONTROLS

    /** sheet name to use */
    @FXML
    private TextField txtSheetName;

    /** number format precision slider */
    @FXML
    private Slider slidePrecision;

    /** label to display number format precision */
    @FXML
    private Label lblPrecision;

    /** checkbox for append mode */
    @FXML
    private CheckBox chkAppend;

    /** checkbox for pubmed linking */
    @FXML
    private CheckBox chkPubmed;

    /** column name for pubmed linking */
    @FXML
    private TextField txtPubmed;

    /** maximum column width */
    @FXML
    private ChoiceBox<MaxWidth> cmbMaxWidth;


    /**
     * This is called when the precision slider changes.  We use it to update
     * the label.
     *
     * @param intValue	new value of slider
     */
    private void updatePrecision(int intValue) {
        this.precision = intValue;
        this.lblPrecision.setText(Integer.toString(intValue));
    }

    @Override
    public boolean isValid() {
        return (this.getOutFile() != null && ! this.txtSheetName.getText().isBlank() && (! chkAppend.isSelected() || this.getOutFile().exists()));
    }

    @Override
    public void apply(KeyedFileMap keyedMap) throws IOException {
        // Get the list of column headers.
        List<String> headers = keyedMap.getHeaders();
        // Now we build the workbook.
        try (CustomWorkbook workbook = this.getWorkbook()) {
            // Get the sheet to hold our output.
            String sheetName = this.txtSheetName.getText();
            workbook.addSheet(sheetName, true);
            // Set the precision and the max width.
            workbook.setPrecision(this.precision);
            workbook.setMaxWidth(this.cmbMaxWidth.getSelectionModel().getSelectedItem().getWidth());
            // This will track the pubmed column.
            int pubmed;
            if (! this.chkPubmed.isSelected())
                pubmed = -1;
            else {
                pubmed = keyedMap.findColumn(this.txtPubmed.getText());
                if (pubmed == -1)
                    throw new IOException("Pubmed column not found.");
            }
            // Create the header row.
            workbook.setHeaders(headers);
            // Next, we fill in the cell values.  We also track how many numbers are in each
            // column.  Note that we aren't using the full-blown ColumnData facility here,
            // since we need to know the string type to format the individual cells regardless
            // of what the entire column looks like, thanks to Excel being nutty about text
            // that looks like numbers.
            int[] intCounts = new int[headers.size()];
            int[] numCounts = new int[headers.size()];
            int[] stringCounts = new int[headers.size()];
            int[] flagCounts = new int[headers.size()];
            for (List<String> record : keyedMap.getRecords()) {
                workbook.addRow();
                int c = 0;
                for (String datum : record) {
                    // Here we determine the cell type.  The key (c == 0) is always a string.
                    if (datum.isBlank())
                        workbook.storeBlankCell();
                    else if (c == pubmed) {
                        // Here we have the pubmed column.  The data is counted as a string
                        // no matter what.  If it is an integer, we hyperlink it.
                        if (ColumnData.INTEGER_PATTERN.matcher(datum).matches()) {
                            workbook.storeCell(Integer.parseInt(datum), String.format(PUBMED_FORMAT, datum), null);
                        } else {
                            workbook.storeCell(Integer.parseInt(datum));
                        }
                        // A pubmed is always treated as a string.
                        stringCounts[c]++;
                    } else if (c >= 1 && ColumnData.DOUBLE_PATTERN.matcher(datum).matches()) {
                        workbook.storeCell(Double.parseDouble(datum));
                        // Count the cell type.  If the column is all integers we use a different
                        // format.
                        if (ColumnData.INTEGER_PATTERN.matcher(datum).matches())
                            intCounts[c]++;
                        numCounts[c]++;
                    } else {
                        workbook.storeCell(datum);
                        // Count the cell type.  If the column is all single-character we use a
                        // different format.
                        if (datum.length() <= 1)
                            flagCounts[c]++;
                        stringCounts[c]++;
                    }
                    c++;
                }
            }
            // Fix up the column formatting.  We auto-size the columns and set the format.
            for (int c = 0; c < headers.size(); c++) {
                // Here we figure out the column format.  The basic formats are integer,
                // floating, flag, and text.  The pubmed column is already handled.
                if (c != pubmed) {
                    if (stringCounts[c] == 0) {
                        // Here we are all numbers.  We use the integer style if we are all integers.
                        if (intCounts[c] >= numCounts[c])
                            workbook.reformatIntColumn(c);
                    } else {
                        // Here there are strings.  We use the flag style if we are all flags.
                        if (flagCounts[c] >= stringCounts[c])
                            workbook.reformatFlagColumn(c);
                    }
                }
                // With the column formatted, we can auto-size it.  We add 512 to the width to
                // provide a extra character space.
                workbook.autoSizeColumn(c);
            }
        }
        this.checkForOpen();
    }

    /**
     * Enable or disable the pubmed column name textbox depending on the state of
     * the checkbox.
     *
     * @param event		event for the checkbox toggle
     */
    @FXML
    public void togglePubmedBox(Event event) {
        this.txtPubmed.setDisable(! this.chkPubmed.isSelected());
    }

    /**
     * @return the user-specified workbook
     *
     * @throws IOException
     */
    private CustomWorkbook getWorkbook() throws IOException {
        CustomWorkbook retVal;
        File bookFile = this.getOutFile();
        if (this.chkAppend.isSelected()) {
            try {
                retVal = CustomWorkbook.load(bookFile);
            } catch (InvalidFormatException e) {
                throw new IOException("Error loading spreadsheet: " + e.getMessage());
            }
        } else {
            retVal = CustomWorkbook.create(bookFile);
        }
        return retVal;
    }

    @Override
    protected String getFileLabel() {
        return "Spreadsheet";
    }

    @Override
    protected ExtensionFilter getFilter() {
        return JoinDialog.EXCEL_FILTER;
    }

    @Override
    protected void initControls() {
        // Default a sheet name.
        this.txtSheetName.setText(this.getParent().getName());
        // Get the initial number format precision.
        this.precision = (int) this.slidePrecision.getValue();
        // Set up a listener for the number format slider.
        this.slidePrecision.valueProperty().addListener((observable, oldVal, newVal) -> {
            this.updatePrecision(newVal.intValue());
        });
        // Default the max width.
        this.cmbMaxWidth.getItems().addAll(MaxWidth.values());
        this.cmbMaxWidth.getSelectionModel().select(MaxWidth.UNLIMITED);
    }

}
