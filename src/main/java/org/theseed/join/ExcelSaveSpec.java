/**
 *
 */
package org.theseed.join;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.regex.Pattern;

import java.awt.Desktop;


import org.apache.poi.hssf.util.HSSFColor;
import org.apache.poi.ss.usermodel.CellStyle;
import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.ss.usermodel.DataFormat;
import org.apache.poi.ss.usermodel.FillPatternType;
import org.apache.poi.ss.usermodel.Font;
import org.apache.poi.ss.usermodel.HorizontalAlignment;
import org.apache.poi.ss.usermodel.IndexedColors;
import org.apache.poi.ss.util.CellRangeAddress;
import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFCellStyle;
import org.apache.poi.xssf.usermodel.XSSFName;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.io.KeyedFileMap;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.stage.FileChooser;
import javafx.stage.FileChooser.ExtensionFilter;

/**
 * This saves a current copy of the output in an excel spreadsheet as a table.
 *
 * @author Bruce Parrello
 *
 */
public class ExcelSaveSpec implements IJoinSpec {

    /** color to use for header */
    private static final short HEAD_COLOR = IndexedColors.INDIGO.getIndex();
    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(ExcelSaveSpec.class);
    /** output file */
    private File outFile;
    /** parent dialog */
    private JoinDialog parent;
    /** display node for this join specification */
    private Node node;
    /** integer data type pattern */
    protected static final Pattern INTEGER_PATTERN = Pattern.compile("\\s*[\\-+]?\\d+");
    /** double data type pattern */
    protected static final Pattern DOUBLE_PATTERN = Pattern.compile("\\s*[\\-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][\\-+]?\\d+)?");

    // CONTROLS

    /** display field for output file name */
    @FXML
    private TextField txtOutputFile;

    /** sheet name to use */
    @FXML
    private TextField txtSheetName;

    /** specification title */
    @FXML
    private Label lblTitle;

    /** checkbox for open-file request */
    @FXML
    private CheckBox chkOpenFile;

    @Override
    public void init(JoinDialog parent, Node node) {
        // Connect the parent dialog.
        this.parent = parent;
        this.node = node;
        // Denote there is no output file.
        this.outFile = null;
        // Default a sheet name.
        this.txtSheetName.setText(parent.getName());
        // If we cannot use the desktop, disable the open-excel button.
        this.chkOpenFile.setVisible(Desktop.isDesktopSupported());
    }

    @Override
    public boolean isValid() {
        return (this.outFile != null && ! this.txtSheetName.getText().isBlank());
    }

    /**
     * Here the user wants to specify the excel output file.
     *
     * @param event		event for button press
     */
    @FXML
    private void selectOutput(ActionEvent event) {
        // Initialize the chooser dialog.
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select Output Spreadsheet");
        chooser.setInitialDirectory(this.parent.getParentDirectory());
        chooser.getExtensionFilters().addAll(new ExtensionFilter("Excel File", "*.xlsx"));
        // Get the file.
        File newOutFile = chooser.showSaveDialog(this.parent.getStage());
        if (newOutFile != null) {
            // We have a new output file.  Save the parent directory for next time.
            this.parent.setParentDirectory(newOutFile.getParentFile());
            // Save the file name.
            this.outFile = newOutFile;
            this.txtOutputFile.setText(this.outFile.getName());
            // Update the state of the parent.
            this.parent.configureButtons();
        }
    }

    @Override
    public void apply(KeyedFileMap keyedMap) throws IOException {
        // Get the list of column headers.
        List<String> headers = keyedMap.getHeaders();
        // Get the number of records.
        int rows = keyedMap.size();
        // Now we build the workbook.
        try (XSSFWorkbook workbook = new XSSFWorkbook()) {
            // Create a new sheet to hold our output.
            String sheetName = this.txtSheetName.getText();
            XSSFSheet newSheet = (XSSFSheet) workbook.createSheet(sheetName);
            // Create the special formatting styles.
            DataFormat format = workbook.createDataFormat();
            short intFmt = format.getFormat("##0");
            short dblFmt = format.getFormat("##0.0000");
            XSSFCellStyle headStyle = workbook.createCellStyle();
            headStyle.setFillForegroundColor(HEAD_COLOR);
            headStyle.setFillPattern(FillPatternType.SOLID_FOREGROUND);
            Font font = workbook.createFont();
            font.setColor(HSSFColor.HSSFColorPredefined.WHITE.getIndex());
            headStyle.setFont(font);
            XSSFCellStyle numStyle = workbook.createCellStyle();
            numStyle.setDataFormat(dblFmt);
            numStyle.setAlignment(HorizontalAlignment.RIGHT);
            XSSFCellStyle intStyle = workbook.createCellStyle();
            intStyle.setDataFormat(intFmt);
            intStyle.setAlignment(HorizontalAlignment.RIGHT);
            XSSFCellStyle flagStyle = workbook.createCellStyle();
            flagStyle.setAlignment(HorizontalAlignment.CENTER);
            XSSFCellStyle cellStyle = workbook.createCellStyle();
            flagStyle.setAlignment(HorizontalAlignment.LEFT);
            // Create the header row.
            XSSFRow row = newSheet.createRow(0);
            for (int c = 0; c < headers.size(); c++) {
                XSSFCell cell = row.createCell(c, CellType.STRING);
                cell.setCellValue(headers.get(c));
                cell.setCellStyle(headStyle);
            }
            // Next, we fill in the cell values.  We also track how many numbers are in each
            // column.
            int[] intCounts = new int[headers.size()];
            int[] numCounts = new int[headers.size()];
            int[] stringCounts = new int[headers.size()];
            int[] flagCounts = new int[headers.size()];
            int r = 1;
            for (List<String> record : keyedMap.getRecords()) {
                row = newSheet.createRow(r);
                int c = 0;
                for (String datum : record) {
                    XSSFCell cell;
                    // Here we determine the cell type.  The key (c == 0) is always a string.
                    if (datum.isBlank())
                        cell = row.createCell(c, CellType.BLANK);
                    else if (c >= 1 && DOUBLE_PATTERN.matcher(datum).matches()) {
                        cell = row.createCell(c, CellType.NUMERIC);
                        cell.setCellValue(Double.parseDouble(datum));
                        // Count the cell type.  If the column is all integers we use a different
                        // format.
                        if (INTEGER_PATTERN.matcher(datum).matches())
                            intCounts[c]++;
                        numCounts[c]++;
                    } else {
                        cell = row.createCell(c, CellType.STRING);
                        cell.setCellValue(datum);
                        // Count the cell type.  If the column is all single-character we use a
                        // different format.
                        if (datum.length() <= 1)
                            flagCounts[c]++;
                        stringCounts[c]++;
                    }
                    c++;
                }
                r++;
            }
            // Fix up the column formatting.  We auto-size the columns and set the format.
            for (int c = 0; c < headers.size(); c++) {
                // Here we figure out the column format.  The basic formats are integer,
                // floating, flag, and text.
                if (stringCounts[c] == 0) {
                    // Here we are all numbers.  We use the integer style if we are all integers.
                    if (intCounts[c] >= numCounts[c])
                        this.formatColumn(newSheet, intStyle, c, rows);
                    else
                        this.formatColumn(newSheet, numStyle, c, rows);
                } else {
                    // Here there are strings.  We use the flag style if we are all flags.
                    if (flagCounts[c] >= stringCounts[c])
                        this.formatColumn(newSheet, flagStyle, c, rows);
                    else
                        this.formatColumn(newSheet, cellStyle, c, rows);
                }
                // With the column formatted, we can auto-size it.  We add 512 to the width to
                // provide a extra character space.
                newSheet.autoSizeColumn(c);
                int oldWidth = newSheet.getColumnWidth(c);
                newSheet.setColumnWidth(c, oldWidth + 512);
            }
            // Set filtering in the top row and name the range.
            CellRangeAddress tableRange = new CellRangeAddress(0, rows, 0, headers.size() - 1);
            newSheet.setAutoFilter(tableRange);
            XSSFName tableName = workbook.createName();
            tableName.setNameName(sheetName);
            String tableRangeAddress = tableRange.formatAsString(sheetName, true);
            tableName.setRefersToFormula(tableRangeAddress);
            // Now write the table out.
            try (FileOutputStream saveStream = new FileOutputStream(this.outFile)) {
                workbook.write(saveStream);
            }
            if (this.chkOpenFile.isSelected()) {
                // Here the user wants to open the file in Excel.
                Desktop myDesktop = Desktop.getDesktop();
                myDesktop.open(this.outFile);
            }
        }
    }

    /**
     * Apply a format to all the cells in a table column.
     *
     * @param sheet		sheet containing the table
     * @param format	formatting style to apply
     * @param c			column index
     * @param rows		number of data rows
     */
    private void formatColumn(XSSFSheet sheet, CellStyle format, int c, int rows) {
        // We don't format the header row, since we want the labels to remain left-aligned
        // and away from the little filter arrows.
        for (int r = 1; r <= rows; r++) {
            XSSFCell cell = sheet.getRow(r).getCell(c);
            cell.setCellStyle(format);
        }
    }

    @Override
    public Node getNode() {
        return this.node;
    }

    /**
     * Here the user wants to delete the file from the file list.
     *
     * @param event		event for button press
     */
    @FXML
    private void deleteFile(ActionEvent event) {
        boolean confirmed = JoinSpec.confirmDelete(this.outFile);
        if (confirmed)
            this.parent.deleteFile(this);
    }

    @Override
    public void setTitle(String title) {
        this.lblTitle.setText(title);
    }

}
