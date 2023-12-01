/**
 *
 */
package org.theseed.dl4j.jfx;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.theseed.dl4j.BalanceColumnFilter;
import org.theseed.dl4j.DistributedOutputStream;
import org.theseed.dl4j.SubsetColumnFilter;
import org.theseed.dl4j.train.ITrainingProcessor;
import org.theseed.io.LineReader;
import org.theseed.io.ParmDescriptor;
import org.theseed.io.ParmFile;
import org.theseed.io.TabbedLineReader;
import org.theseed.jfx.BaseController;
import org.theseed.jfx.MovableController;

import dl4j.utils.ConversionStream;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.Spinner;
import javafx.scene.control.TextField;
import javafx.scene.control.TextFormatter;
import javafx.stage.FileChooser;
import javafx.stage.FileChooser.ExtensionFilter;
import javafx.util.converter.IntegerStringConverter;
import javafx.beans.value.ObservableValue;

/**
 * This controller manages the process of creating a training file from external data.  There must
 * already be a training.tbl file in the model directory, but it can contain just headers.  Normally, the
 * old file will be backed up and destroyed; however, an option is provided to specify an alternate location
 * for the result.
 *
 * For a classification model, the data will be balanced.  For a regression model, the data will be scrambled.
 * If the input file is in a different format, it will be converted.
 *
 * @author Bruce Parrello
 *
 */
public class BalanceInput extends MovableController {

    // FIELDS
    /** model directory */
    private File modelDirectory;
    /** input file */
    private File inputFile;
    /** output file */
    private File outputFile;
    /** parameter file controller */
    private ParmFile parms;
    /** parameter file name */
    private File parmFile;
    /** current model processor */
    private ITrainingProcessor processor;
    /** impact file rating set */
    private List<String> impactList;

    // CONTROLS

    /** choice box for input format */
    @FXML
    private ChoiceBox<InputFormat> cmbFormat;

    /** text field for displaying input file name */
    @FXML
    private TextField txtInputFile;

    /** text field for displaying output file name */
    @FXML
    private TextField txtOutputFile;

    /** button to start the process */
    @FXML
    private Button btnRun;

    /** checked if we need to add an ID field */
    @FXML
    private CheckBox chkMakeIDs;

    /** choice box for selecting label of interest */
    @FXML
    private ChoiceBox<String> cmbLabel;

    /** label for column restrictor */
    @FXML
    private Label labelColumns;

    /** text box for column restrictor */
    @FXML
    private TextField textColumns;

    /** text box for missing-column default value */
    @FXML
    private TextField txtDefaultValue;

    /** check box for enabling column restrictor */
    @FXML
    private CheckBox checkColumns;

    /** combo box for selecting scramble column */
    @FXML
    private ChoiceBox<String> cmbScramble;

    /** check box for enabling row balancing */
    @FXML
    private Spinner<Double> spinBalanced;

    public BalanceInput() {
        super(200, 200);
    }

    @Override
    public String getIconName() {
        return "job-16.png";
    }

    @Override
    public String getWindowTitle() {
        return "Create Balanced Training File";
    }

    /**
     * Use the model directory to prime the input file search.
     *
     * @param	modelDir	input model directory
     * @param	parmFile	relevant parameter file
     * @param	type		model type
     *
     * @throws IOException
     */
    public void init(File modelDir, File parmFile, ITrainingProcessor processor) throws IOException {
        // Save the parameters and the model directory.
        this.modelDirectory = modelDir;
        this.parmFile = parmFile;
        this.parms = new ParmFile(parmFile);
        this.processor = processor;
        // Initialize the format combo box.
        this.cmbFormat.getItems().addAll(InputFormat.values());
        this.cmbFormat.getSelectionModel().select(InputFormat.PATRIC);
        // Set up the label combo box.
        List<String> labelCols = processor.getLabelCols();
        this.cmbLabel.getItems().addAll(labelCols);
        this.cmbLabel.getSelectionModel().select(0);
        // Set up the scrambler combo box.  We need to read the headers from training.tbl.
        List<String> allCols = this.getAllCols();
        allCols.add(0, "(none)");
        this.cmbScramble.getItems().addAll(allCols);
        this.cmbScramble.getSelectionModel().select(0);
        // Find out if we should generate IDs.
        this.chkMakeIDs.setSelected(this.parms.get("id").isCommented());
        // Specify a default value for missing columns.
        this.txtDefaultValue.setText("0.0");
        // Set up the output file.
        this.outputFile = new File(this.modelDirectory, "training.tbl");
        this.txtOutputFile.setText(this.outputFile.getName());
        // Check for an impact file.
        File impactFile = new File(modelDir, "impact.tbl");
        boolean canRestrict = impactFile.canRead();
        this.checkColumns.setVisible(canRestrict);
        this.labelColumns.setVisible(canRestrict);
        this.textColumns.setVisible(canRestrict);
        // Put in the text control validator.
        if (canRestrict) {
            // Set up the text control to default to a reduction by half, and for numbers only.
            // First, we need to read the impact list.
            int nonZero = 0;
            this.impactList = new ArrayList<String>(100);
            try (TabbedLineReader reader = new TabbedLineReader(impactFile)) {
                int idCol = reader.findField("col_name");
                int valCol = reader.findField("info_gain");
                for (TabbedLineReader.Line line : reader) {
                    this.impactList.add(line.get(idCol));
                    double value = line.getDouble(valCol);
                    if (value > 0.0) nonZero++;
                }
            }
            int defaultValue = Math.min((this.impactList.size() + 1) / 2, nonZero);
            TextFormatter<Integer> formatter = new TextFormatter<>(
                    new IntegerStringConverter(), defaultValue,
                    c -> Pattern.matches("\\d*", c.getText()) ? c : null );
            this.textColumns.setTextFormatter(formatter);
            // Disable the text control until the checkbox is checked.
            this.textColumns.setDisable(true);
            // Set up the checkbox to enable/disable depending on whether or not it is checked.
            this.checkColumns.selectedProperty().addListener(
                    (ObservableValue<? extends Boolean> ov, Boolean oldV, Boolean newV) ->
                    { this.textColumns.setDisable(! newV); } );
        }
        // Denote we are not ready to run.
        btnRun.setDisable(true);
    }

    /**
     * @return a list of all the column headers in the current training file
     *
     * @throws IOException
     */
    private List<String> getAllCols() throws IOException {
        List<String> retVal;
        File trainingFile = new File(this.modelDirectory, "training.tbl");
        try (TabbedLineReader trainingStream = new TabbedLineReader(trainingFile)) {
            // We need a buffer variable that can be declared final for the lambda expression below.
            final List<String> buffer = new ArrayList<String>(trainingStream.size());
            Arrays.stream(trainingStream.getLabels()).forEach(x -> buffer.add(x));
            retVal = buffer;
        }
        return retVal;
    }

    /**
     * Select the input file.
     *
     * @param event		event descriptor
     */
    public void selectInput(Event event) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select Raw Input File");
        chooser.setInitialDirectory(this.modelDirectory);
        // We add a special filter for the preferred extension of the selected format.
        ExtensionFilter specialFilter = this.cmbFormat.getValue().getFilter();
        chooser.getExtensionFilters().addAll(specialFilter,
                new ExtensionFilter("All Files", "*.*"));
        // Get the file.
        File retVal = chooser.showOpenDialog(this.getStage());
        if (retVal != null) {
            this.inputFile = retVal;
            txtInputFile.setText(this.inputFile.getName());
            btnRun.setDisable(false);
        }
    }

    /**
     * Select the output file.
     *
     * @param event		event descriptor
     */
    public void selectOutput(Event event) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select Output File");
        chooser.setInitialDirectory(this.outputFile.getParentFile());
        // We add a special filter for the preferred extension of the selected format.
        chooser.getExtensionFilters().addAll(new ExtensionFilter("Training Files", "*.tbl"),
                new ExtensionFilter("All Files", "*.*"));
        // Get the file.
        File retVal = chooser.showSaveDialog(this.getStage());
        if (retVal != null) {
            this.outputFile = retVal;
            txtOutputFile.setText(retVal.getName());
        }
    }

    /**
     * Create the balanced input file.  This randomizes the data in such a way that different
     * output values are evenly distributed.
     *
     * @param event		event descriptor
     */
    public void runBalancer(Event event) {
        // Compute the backup file name.
        File backupFile = new File(this.modelDirectory, this.outputFile.getName() + ".bak");
        // Get the input type.
        InputFormat type = this.cmbFormat.getValue();
        // We use this to determine if we need to restore from backup.
        boolean restoreNeeded = false;
        // Compute the stream to use for the input.
        File trainingFile = new File(this.modelDirectory, "training.tbl");
        try (InputStream sourceStream = type.getInStream(this.inputFile, trainingFile, txtDefaultValue.getText());
                LineReader inStream = new LineReader(sourceStream)) {
            // Get the headers.
            String[] headers = type.getHeaders(inStream, this.outputFile);
            // Get the column filter.
            BalanceColumnFilter filter;
            if (checkColumns.isSelected())
                filter = new SubsetColumnFilter(this.impactList, (int) this.textColumns.getTextFormatter().getValue(),
                        this.processor.getMetaList(), this.processor.getLabelCols());
            else
                filter = new BalanceColumnFilter.All();
            // Make a safety copy of the training file.
            if (this.outputFile.exists()) {
                FileUtils.copyFile(this.outputFile, backupFile);
                restoreNeeded = true;
            }
            // Process the input to produce the output.
            boolean ok = this.balanceInput(this.outputFile, type, inStream, headers, filter);
            // End the dialog.
            if (ok) {
                this.getStage().close();
                restoreNeeded = false;
                BaseController.messageBox(Alert.AlertType.INFORMATION, "Input Conversion", this.outputFile.getPath() + " has been updated.");
            }
        } catch (IOException e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Converting Input", e.toString());
        }
        if (restoreNeeded) {
            // Here we must restore the training.tbl file from backup.
            try {
                FileUtils.copyFile(backupFile, this.outputFile);
            } catch (IOException e) {
                // Here we couldn't restore the backup, which is really bad.
                BaseController.messageBox(Alert.AlertType.ERROR, "Error Restoring Output File",
                        e.toString());
            }
        }
    }

    /**
     * Produce the output by balancing the input.
     *
     * @param trainingFile	name of the output training file
     * @param type			type of input
     * @param inStream		input stream
     * @param headers		headers from the training file
     * @param filter		column filter
     *
     * @return TRUE if successful, else FALSE
     */
    private boolean balanceInput(File trainingFile, InputFormat type, Iterator<String> inStream,
            String[] headers, BalanceColumnFilter filter) {
        boolean retVal = false;
        // Here we must build the final header list.  We also need a map of input columns to output columns.
        boolean idColFlag = this.chkMakeIDs.isSelected();
        int[] columnMap = this.computeColumnMap(headers, filter, idColFlag);
        // Compute the final column count.
        int outColumns = columnMap.length;
        // Create the actual header list.
        String[] actualHeaders = new String[outColumns];
        int iCol = 0;
        if (idColFlag) actualHeaders[iCol++] = "id";
        while (iCol < outColumns) {
            actualHeaders[iCol] = headers[columnMap[iCol]];
            iCol++;
        }
        // Initialize the counter used to generate IDs.
        int idCounter = 1;
        try (DistributedOutputStream outStream = DistributedOutputStream.create(trainingFile, this.processor, this.cmbLabel.getValue(), actualHeaders)) {
            // If we are doing class-balancing, turn on the switch.
            outStream.setBalanced(this.spinBalanced.getValue());
            // Initially, we stash all the output lines in here.
            List<String[]> lines = new ArrayList<String[]>(1000);
            while (inStream.hasNext()) {
                String[] items = type.getLine(inStream);
                String[] outItems = new String[outColumns];
                // Only proceed if there is data in the line.  A lot of datasets out there have blanks in them.
                if (items.length > 0) {
                    for (int i = 0; i < outColumns; i++) {
                        if (columnMap[i] < 0)
                            outItems[i] = String.format("row-%04d", idCounter++);
                        else
                            outItems[i] = items[columnMap[i]];
                    }
                    lines.add(outItems);
                }
            }
            // If we are scrambling, Do it here.
            String scrambleCol = this.cmbScramble.getValue();
            int iScramble = -1;
            for (int i = 0; i < actualHeaders.length && iScramble < 0; i++) {
                if (actualHeaders[i].contentEquals(scrambleCol))
                    iScramble = i;
            }
            if (iScramble >= 0)
                this.scrambleColumn(lines, iScramble);
            // Now we write the output.
            for (String[] outItems : lines)
                outStream.write(outItems);
            // Make the necessary updates to the parm file.
            this.updateParmFile(lines.size());
            // Denote we've succeeded.
            retVal = true;
        } catch (IOException e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Writing Output", e.toString());
        }
        return retVal;
    }

    /**
     * Scramble the order of the items in the specified column.
     *
     * @param lines			list of output lines, each represented as an array of values
     * @param iScramble		index of the column to scramble
     */
    private void scrambleColumn(List<String[]> lines, int iScramble) {
        // Get the scrambling column into a list and shuffle it.
        List<String> column = lines.stream().map(x -> x[iScramble]).collect(Collectors.toList());
        Collections.shuffle(column);
        // Store the scrambled results back over the top of the scramble column in the input list.
        for (int i = 0; i < lines.size(); i++)
            lines.get(i)[iScramble] = column.get(i);
    }

    /**
     * This method creates a map showing what incoming column goes in each output column for the balanced
     * dataset.  The special code -1 is used for the ID column.
     *
     * @param headers	incoming column headers
     * @param filter	filter used to decide which columns to keep
     *
     * @return an array showing the input column index for each output column.
     */
    private int[] computeColumnMap(String[] headers, BalanceColumnFilter filter, boolean idColFlag) {
        List<Integer> numberMap = new ArrayList<Integer>(headers.length);
        if (idColFlag) numberMap.add(-1);
        for (int i = 0; i < headers.length; i++) {
            if (filter.allows(headers[i]))
                numberMap.add(i);
        }
        return numberMap.stream().mapToInt(i -> i).toArray();
    }

    /**
     * Update the parameter file.  We adjust the testing set size, and possible modify the ID column.
     *
     * @param dataCount		number of records written
     *
     * @throws IOException
     */
    public void updateParmFile(int dataCount) throws IOException {
        // Update the testing set size.
        ParmDescriptor testSizeParm = this.parms.get("testSize");
        int testSizeCount = dataCount / 10;
        if (testSizeCount < 1) testSizeCount = 1;
        testSizeParm.setValue(Integer.toString(testSizeCount));
        // Update the sample size, if any.
        ParmDescriptor numExamplesParm = this.parms.get("sampleSize");
        if (numExamplesParm != null)
            numExamplesParm.setValue(Integer.toString(testSizeCount * 2));
        // Now we add the ID column, if any.
        if (this.chkMakeIDs.isSelected()) {
            ParmDescriptor colParm = this.parms.get("meta");
            colParm.setCommented(false);
            String newValue = colParm.getValue();
            if (newValue.isEmpty())
                newValue = "id";
            else {
                String[] parts = StringUtils.split(newValue, ',');
                if (! Arrays.stream(parts).anyMatch(x -> x.contentEquals("id")))
                    newValue = "id," + newValue;
            }
            colParm.setValue(newValue);
            colParm = this.parms.get("id");
            colParm.setCommented(false);
            colParm.setValue("id");
        }
        // Save the parameter file.
        this.parms.save(this.parmFile);
    }

    /**
     * This enum handles the differences in input handling between the various formats.
     *
     * KERAS 	comma-delimited, no headers
     * PATRIC	tab-delimited, with headers
     */
    private static enum InputFormat {
        KERAS {
            @Override
            public String[] getLine(Iterator<String> inStream) throws IOException {
                String[] retVal = StringUtils.stripAll(StringUtils.splitPreserveAllTokens(inStream.next(), ','), "\"");
                return retVal;
            }

            @Override
            public String[] getHeaders(Iterator<String> inStream, File trainingFile) throws IOException {
                String[] retVal = null;
                try (LineReader hdrStream = new LineReader(trainingFile)) {
                    retVal = StringUtils.splitPreserveAllTokens(hdrStream.next(), '\t');
                }
                return retVal;
            }

            @Override
            protected ExtensionFilter getFilter() {
                return new ExtensionFilter("Comma-separated file", "*.csv", "*.data");
            }

            @Override
            public InputStream getInStream(File sourceFile, File trainingFile, String defaultV)
                    throws FileNotFoundException {
                return new FileInputStream(sourceFile);
            }
        },
        SIMILAR {
            @Override
            public String[] getLine(Iterator<String> inStream) throws IOException {
                return StringUtils.splitPreserveAllTokens(inStream.next(), '\t');
            }

            @Override
            protected ExtensionFilter getFilter() {
                return new ExtensionFilter("Training file", "*.tbl");
            }

            @Override
            public String[] getHeaders(Iterator<String> inStream, File trainingFile) throws IOException {
                return StringUtils.splitPreserveAllTokens(inStream.next(), '\t');
            }

            @Override
            public InputStream getInStream(File sourceFile, File trainingFile, String defaultV)
                    throws IOException {
                return new ConversionStream(sourceFile, trainingFile, defaultV);
            }

        },
        PATRIC {
            @Override
            public String[] getLine(Iterator<String> inStream) throws IOException {
                return StringUtils.splitPreserveAllTokens(inStream.next(), '\t');
            }

            @Override
            public String[] getHeaders(Iterator<String> inStream, File trainingFile) throws IOException {
                return StringUtils.splitPreserveAllTokens(inStream.next(), '\t');
            }

            @Override
            protected ExtensionFilter getFilter() {
                return new ExtensionFilter("Tab-delimited file", "*.tbl", "*.txt", "*.tsv");
            }

            @Override
            public InputStream getInStream(File sourceFile, File trainingFile, String defaultV)
                    throws IOException {
                return new FileInputStream(sourceFile);
            }
        };

        /**
         * @return the fields of the input line as an array
         *
         * @param inStream	iterator from which the next input line should be retrieved
         */
        public abstract String[] getLine(Iterator<String> inStream) throws IOException;

        /**
         * @return the special filename filter for this input type
         */
        protected abstract ExtensionFilter getFilter();

        /**
         * @return the array of headers for this operation
         *
         * @param inStream		iterator from which the first input line should be retrieved
         * @param trainingFile	original training file, used for headers in KERAS mode
         */
        public abstract String[] getHeaders(Iterator<String> inStream, File trainingFile) throws IOException;

        /**
         * @return the input stream for this operation
         *
         * @param sourceFile	source file name
         * @param trainingFile	target training file name
         * @param defaultV		default value for missing columns
         *
         * @throws IOException
         */
        public abstract InputStream getInStream(File sourceFile, File trainingFile, String defaultV)
                throws IOException;
    }


}
