/**
 *
 */
package org.theseed.dl4j.jfx;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.theseed.jfx.ColumnAnalysis;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;
import org.theseed.io.TabbedLineReader;
import org.theseed.jfx.BaseController;
import org.theseed.jfx.DistributionAnalysis;
import org.theseed.jfx.ResizableController;
import org.theseed.jfx.SpreadColumnAnalysis;
import org.theseed.jfx.StatisticsAnalysis;

import javafx.beans.property.SimpleStringProperty;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.SelectionMode;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableColumn.CellDataFeatures;
import javafx.scene.control.TableView;
import javafx.scene.control.TitledPane;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.TilePane;
import javafx.util.Callback;

/**
 * This class displays the contents of the training file.  The file is tab-delimited, so it is loaded
 * into a table view.
 *
 * @author Bruce Parrello
 *
 */
public class TrainingView extends ResizableController {

    /**
     * This is a simple nested class to display the appropriate field of a line in a table cell.
     */
    public class CellDisplayer implements Callback<CellDataFeatures<String[], String>, ObservableValue<String>> {

        private int idx;

        /**
         * Construct this object to display the specified table column's fields.
         *
         * @param idx		table column index
         */
        public CellDisplayer(int idx) {
            this.idx = idx;
        }

        @Override
        public ObservableValue<String> call(CellDataFeatures<String[], String> param) {
            return new SimpleStringProperty(param.getValue()[idx]);
        }

    }

    // FIELDS
    /** list of data lines from the file */
    private ObservableList<String[]> lineBuffer;
    /** headers from the file */
    private String[] headers;
    /** label column index (-1 if none) */
    private int labelIdx;
    /** list of label names */
    private List<String> labelNames;
    /** projection of input colums */
    private INDArray inputProjection;
    /** array of row IDs */
    private String[] rowIDs;
    /** map of output column names to real value arrays */
    private Map<String, double[]> realColumns;
    /** map of output column names to class value arrays */
    private Map<String, int[]> classColumns;
    /** map of output column names to class name sets */
    private Map<String, String[]> classNames;
    /** minimum value of output range */
    private double minValue;
    /** maximum value of output range */
    private double maxValue;
    /** distance between middle and edge of output range */
    private double valueRange;
    /** midpoint between minimum and maximum */
    private double midValue;


    // CONTROLS

    /** name of training file */
    @FXML
    private Label lblFileName;

    /** table to display the data */
    @FXML
    private TableView<String[]> tblView;

    /** list of columns to select */
    @FXML
    private ListView<String> lstColumns;

    /** output panel for analysis graphs */
    @FXML
    private Pane paneResults;

    /** button for displaying spread chart */
    @FXML
    private Button spreadButton;

    /** combo box for selecting label */
    @FXML
    private ComboBox<String> cmbLabel;

    /** graph control */
    @FXML
    private ScatterChart<Double, Double> scatterGraph;

    /** tile pane for legend */
    @FXML
    private TilePane paneLegend;

    /**
     * Position and size this window.
     */
    public TrainingView() {
        super(200, 200, 1000, 750);
    }

    @Override
    public String getIconName() {
        return "job-16.png";
    }

    @Override
    public String getWindowTitle() {
        return "Training File Display";
    }

    /**
     * Initialize this page.  All of the columns will be created in the table and the data lines read in.
     *
     * @param trainingFile		file name of the incoming training file
     * @param labelIdx			index of the label column (-1 if none)
     * @param labelNames		name of the labels
     * @param nonInputCols		names of the non-input columns
     * @param idColName 		name of the ID column
     */
    public void init(File trainingFile, int labelIdx, List<String> labelNames,
            List<String> nonInputCols, String idColName) {
        // Save the label column index and the label names.
        this.labelIdx = labelIdx;
        this.labelNames = labelNames;
        // Configure the spread button.
        if (this.labelIdx < 0)
            spreadButton.setVisible(false);
        // Save the file name for display.
        this.lblFileName.setText(trainingFile.getAbsolutePath());
        try (TabbedLineReader inStream = new TabbedLineReader(trainingFile)) {
            // Use the header to create the columns.
            this.headers = inStream.getLabels();
            for (int i = 0; i < headers.length; i++) {
                TableColumn<String[], String> newCol = new TableColumn<>(this.headers[i]);
                newCol.setCellValueFactory(new CellDisplayer(i));
                this.tblView.getColumns().add(newCol);
            }
            // Now loop through the input, storing each line as a table row.
            this.lineBuffer = FXCollections.observableArrayList();
            for (TabbedLineReader.Line line : inStream)
                this.lineBuffer.add(line.getFields());
            this.tblView.setItems(this.lineBuffer);
            // Populate the list on the stats panel.
            this.lstColumns.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);
            this.lstColumns.setItems(FXCollections.observableArrayList(this.headers));
            // Now we need to set up the scatter graph structures.
            this.parseInput(nonInputCols, idColName, this.lineBuffer, this.headers);
            ObservableList<String> columns = FXCollections.observableList(nonInputCols);
            this.cmbLabel.setItems(columns);
            // Remove the axis labels.
            this.scatterGraph.getXAxis().setTickLabelsVisible(false);
            this.scatterGraph.getYAxis().setTickLabelsVisible(false);
        }  catch (IOException e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Readining Training File", e.toString());
        }
    }

    /**
     * This column prepares the input for the visualization graph.  We need a projected INDArray
     * for the X and Y coordinates of the points, an array of the ID strings, and arrays of all
     * the non-input column values.  If a non-input column is all numbers, they will be converted
     * and stored as real numbers.  Strings are sorted into classes and stored as class indices and
     * names.  So, for string columns, we have a scatter graph of index-colored points.  For number
     * columns, we have a scatter graph of gradient-colored points.  It is all very complicated.
     *
     * @param nonInputCols		list of the names of the non-input columns (except the ID column)
     * @param idColName			name of the ID column
     * @param inputLines		list of input lines, stored as string arrays
     * @param headers			array of column names
     */
    private void parseInput(List<String> nonInputCols, String idColName, List<String[]> inputLines,
            String[] headers) {
        // Generate the main INDArray for the inputs.
        int cols = headers.length - 1 - nonInputCols.size();
        int rows = inputLines.size();
        INDArray inputs = Nd4j.create(rows, cols);
        // This will be our current output position in the INDArray.
        int oCol = 0;
        // Generate the column-tracking structures.
        int hashSize = nonInputCols.size() * 2;
        this.classColumns = new HashMap<String, int[]>(hashSize);
        this.classNames = new HashMap<String, String[]>(hashSize);
        this.realColumns = new HashMap<String, double[]>(hashSize);
        // This will track the current column index in the input-value matrix.
        // Now we loop through the columns one at a time.
        for (int c = 0; c < headers.length; c++) {
            // Get a final version of the column index for streaming, and the column name.
            final int col = c;
            String colName = headers[c];
            // Determine the column type.
            if (colName.contentEquals(idColName)) {
                // Here we have the ID column.
                this.rowIDs = inputLines.stream().map(x -> x[col]).toArray(String[]::new);
            } else if (! nonInputCols.contains(colName)) {
                // Here we have an input column.
                for (int r = 0; r < rows; r++) {
                    double item = Double.valueOf(inputLines.get(r)[c]);
                    inputs.put(r, oCol, item);
                }
                oCol++;
            } else {
                // Here we have a possible result column.  It can contain classes or real values.
                // We check for real values first.
                double[] column = this.parseColumn(inputLines, col);
                if (column != null)
                    this.realColumns.put(colName, column);
                else {
                    // Here it must be a class column.
                    List<String> classList = new ArrayList<String>();
                    int[] classes = this.parseColumn(inputLines, col, classList);
                    this.classColumns.put(colName, classes);
                    String[] classArray = classList.stream().toArray(String[]::new);
                    this.classNames.put(colName, classArray);
                }
            }
        }
        // Project the inputs onto 2 dimensions.
        INDArray workMatrix = inputs.dup();
        INDArray projector = PCA.pca_factor(workMatrix, 2, false);
        this.inputProjection = inputs.mmul(projector);
    }

    /**
     * Convert an input column to a list of class indexes.  In this case, the number of different
     * string values in the input column is expected to be small.  The provided class list will be
     * filled with the values found, and the indices into the list stored in the output array.
     *
     * @param inputLines	input lines to parse
     * @param col			index of the input column of interest
     * @param classList		output list for the classes found
     *
     * @return an integer array containing the index of the class found in each row
     */
    private int[] parseColumn(List<String[]> inputLines, int col, List<String> classList) {
        int rows = inputLines.size();
        int[] retVal = new int[rows];
        for (int r = 0; r < rows; r++) {
            String datum = inputLines.get(r)[col];
            int idx = classList.indexOf(datum);
            if (idx >= 0)
                retVal[r] = idx;
             else {
                // Here we have a new class.
                retVal[r] = classList.size();
                classList.add(datum);
            }
        }
        return retVal;
    }

    /**
     * This method will attempt to parse the specified input column into an array of real
     * numbers.  If it fails, it will return NULL, indicating the column is a class column,
     * not a numeric column.
     *
     * @param inputLines	input data lines
     * @param col			index of the input column of interest
     *
     * @return an array of real numbers, containing the number in each row of the column, or NULL
     */
    private double[] parseColumn(List<String[]> inputLines, int col) {
        int rows = inputLines.size();
        double[] retVal = new double[rows];
        // Loop through the rows.  If we find a bad data item (which will usually be the first),
        // we null the return variable, stopping the loop.
        for (int r = 0; r < rows && retVal != null; r++) {
            try {
                retVal[r] = Double.valueOf(inputLines.get(r)[col]);
            } catch (NumberFormatException e) {
                // Here we have an invalid number, so we stop the conversions.
                retVal = null;
            }
        }
        return retVal;
    }

    /**
     * Here the user wants to do a distribution analysis of the selected columns.
     *
     * @param event		event descriptor
     */
    @FXML
    public void analyzeColumns(ActionEvent event) {
        ColumnAnalysis analyzer = new DistributionAnalysis(this.lineBuffer, this.labelIdx);
        this.processColumns(analyzer);
    }

    /**
     * Here the user wants mean and standard deviation statistics on the selected columns.
     *
     * @param event		event descriptor
     */
    @FXML
    public void computeStatistics(ActionEvent event) {
        ColumnAnalysis analyzer = new StatisticsAnalysis(this.lineBuffer, this.labelIdx);
        this.processColumns(analyzer);
    }

    /**
     * Here the user wants a display of the classification spread on the selected columns.
     *
     * @param event		event descriptor
     */
    @FXML
    public void displaySpread(ActionEvent event) {
        ColumnAnalysis analyzer = new SpreadColumnAnalysis(this.lineBuffer, this.labelIdx, this.labelNames);
        this.processColumns(analyzer);
    }

    /**
     * Here the user wants to erase all the results displayed in the result panel.
     *
     * @param event		event descriptor
     */
    @FXML
    public void clearResults(ActionEvent event) {
        this.paneResults.getChildren().clear();
    }

    /**
     * Produce results for the currently-selected columns and add them to the result panel.
     *
     * @param analyzer		analyzer to apply for producing the results.
     */
    private void processColumns(ColumnAnalysis analyzer) {
        // Get the selected column indices.
        ObservableList<Integer> selected = this.lstColumns.getSelectionModel().getSelectedIndices();
        for (int selectIdx : selected) {
            TitledPane result = analyzer.getDisplay(this.headers[selectIdx], selectIdx);
            this.paneResults.getChildren().add(result);
        }
        // De-select everything.
        this.lstColumns.getSelectionModel().clearSelection();
    }

    /**
     * Display the scatter plot for the selected column.
     */
    @FXML
    protected void displayGraph() {
        // Erase the custom legend.
        this.paneLegend.getChildren().clear();
        // Process according to the type of column selected.
        String colName = this.cmbLabel.getSelectionModel().getSelectedItem();
        if (colName == null)
            BaseController.messageBox(AlertType.WARNING, "Visualization", "No data column is selected.");
        else if (this.classColumns.containsKey(colName))
            this.displayClassGraph(colName);
        else
            this.displayNumberGraph(colName);
    }

    /**
     * This method will display a scatter plot in a single series, with the color of each dot indicating
     * where it is in the range of output values, green for the minimum, red for the maximum, and yellow
     * for the midpoint.
     *
     * @param colName	name of the selected output column
     */
    private void displayNumberGraph(String colName) {
        double[] values = this.realColumns.get(colName);
        var points = new XYChart.Series<Double, Double>();
        ObservableList<XYChart.Series<Double, Double>> allSeries = FXCollections.observableArrayList();
        points.setName(colName);
        this.minValue = values[0];
        this.maxValue = values[0];
        for (int r = 0; r < values.length; r++) {
            var point = createPoint(r);
            points.getData().add(point);
            if (values[r] < this.minValue) this.minValue = values[r];
            if (values[r] > this.maxValue) this.maxValue = values[r];
        }
        // Store the points in the graph.
        allSeries.add(points);
        this.scatterGraph.setData(allSeries);
        this.valueRange = (this.maxValue - this.minValue) / 2;
        this.midValue = (this.maxValue + this.minValue) / 2;
        // Now go through the points, setting the colors and tooltips.
        for (int r = 0; r < values.length; r++) {
            var point = (StackPane) points.getData().get(r).getNode();
            // Set up the tooltip.
            var tooltip = new Tooltip(String.format("%s %1.4f", this.rowIDs[r], values[r]));
            Tooltip.install(point, tooltip);
            if (this.minValue < this.maxValue) {
                String style = this.getPointStyle(values[r]);
                point.setStyle(style);
                point.getStyleClass().clear();
            }
        }
        // Finally, we need to create the legend.  We do 5 points: minimum, low, middle, high, maximum.
        double width = this.valueRange / 2;
        for (double val = this.minValue; val <= this.maxValue; val += width) {
            String text = String.format("%6.4f", val);
            Region point = new Region();
            String style = this.getPointStyle(val);
            point.setStyle(style);
            Label legendItem = new Label(text, point);
            this.paneLegend.getChildren().add(legendItem);
        }
    }

    /**
     * @return the style string for a point with the specified value
     *
     * @param val	output value of interest
     */
    public String getPointStyle(double val) {
        int red;
        int green;
        if (val > this.midValue) {
            red = 200;
            green = (int) Math.round((this.maxValue - val) * 200 / this.valueRange);
        } else {
            red = (int) Math.round((val - this.minValue) * 200 / this.valueRange);
            green = 200;
        }
        String style = String.format("-fx-background-radius: 4px; -fx-padding: 4px; -fx-background-color: #%02X%02X00;", red, green);
        return style;
    }

    /**
     * @return the data point location for the specified data row
     *
     * @param r		data row of interest
     */
    public XYChart.Data<Double, Double> createPoint(int r) {
        return new XYChart.Data<Double, Double>(this.inputProjection.getDouble(r, 0),
                this.inputProjection.getDouble(r, 1));
    }

    /**
     * This method will display a classification-based scatter plot.  Each class will be set up as a different
     * series, and a legend across the bottom will identify the color for each series.
     *
     * @param colName	name of the selected output column
     */
    private void displayClassGraph(String colName) {
        int[] classNums = this.classColumns.get(colName);
        String[] classNameArray = this.classNames.get(colName);
        ObservableList<XYChart.Series<Double, Double>> allSeries = FXCollections.observableArrayList();
        for (int i = 0; i < classNameArray.length; i++) {
            var points = new XYChart.Series<Double, Double>();
            points.setName(classNameArray[i]);
            allSeries.add(points);
        }
        // Now we loop through the data rows, adding each point to the appropriate series.
        for (int r = 0; r < classNums.length; r++) {
            var point = this.createPoint(r);
            point.setExtraValue(this.rowIDs[r]);
            allSeries.get(classNums[r]).getData().add(point);
        }
        // Store the points in the graph.
        this.scatterGraph.setData(allSeries);
        // Now do the tooltips.
        for (XYChart.Series<Double, Double> points : allSeries) {
            for (XYChart.Data<Double, Double> point : points.getData()) {
                var tooltip = new Tooltip((String) point.getExtraValue());
                Tooltip.install(point.getNode(), tooltip);
            }
        }
        // Here we create the legend.  Each series gets a styled point with its own name.
        for (int i = 0; i < allSeries.size(); i++) {
            var points = allSeries.get(i).getData();
            if (points.size() > 0) {
                // We have points in this series, so we can do a legend item.  Copy the
                // style of the first plotted point.
                var point0 = points.get(0).getNode();
                Region symbol = new Region();
                symbol.getStyleClass().addAll(point0.getStyleClass());
                // Form it into a label with the class name.
                Label legendItem = new Label(classNameArray[i], symbol);
                this.paneLegend.getChildren().add(legendItem);
            }
        }
    }

}
