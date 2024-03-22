/**
 *
 */
package org.theseed.dl4j.jfx;

import java.util.Collections;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.theseed.jfx.Prediction;
import org.theseed.jfx.Stat;
import org.theseed.stats.RatingList;

import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TableView;
import javafx.scene.input.Clipboard;
import javafx.scene.input.ClipboardContent;

/**
 * This is the controller for the scatter graph that displays expected vs. predicted values of a regression model.
 * The scatter graph is a useful visual guide to the model's accuracy.
 *
 * @author Bruce Parrello
 *
 */
public class ScatterGraph extends ValidationDisplayReport {

    // FIELDS
    /** index of the current label of interest */
    private int labelIdx;
    /** training series */
    private XYChart.Series<Double, Double> trainingPoints;
    /** testing series */
    private XYChart.Series<Double, Double> testingPoints;
    /** outlier list */
    private RatingList<Prediction> outliers;
    /** outlier table items */
    private ObservableList<Prediction> outlierItems;
    /** recommended maximum outliers */
    private static final int MAX_OUTLIERS = 20;

    // CONTROLS

    /** combo box for selecting label */
    @FXML
    private ComboBox<String> cmbLabel;

    /** graph control */
    @FXML
    private ScatterChart<Double, Double> scatterGraph;

    /**
     * The constructor simply calls through to the base class.
     */
    public ScatterGraph() {
        super();
    }

    /**
     * Initialize the display controls.
     *
     * @param labels	list of the labels for the parent model
     */
    @Override
    public void init(List<String> labels) {
        // Fill in the combo box for choosing the labels.
        ObservableList<String> labelList = FXCollections.observableArrayList(labels);
        this.cmbLabel.setItems(labelList);
        this.cmbLabel.getSelectionModel().select(0);
        // Create the two series sets.  We put a dummy point in each to insure the legend plots correctly.
        // The legend does not update when we update the series values.
        this.trainingPoints = new XYChart.Series<Double, Double>();
        this.trainingPoints.setName("training");
        this.trainingPoints.getData().add(new XYChart.Data<Double, Double>(0.0, 0.0));
        this.testingPoints = new XYChart.Series<Double, Double>();
        this.testingPoints.setName("testing");
        this.testingPoints.getData().add(new XYChart.Data<Double, Double>(0.0, 0.0));
        // Put the data series into the graph.
        this.scatterGraph.getData().add(this.trainingPoints);
        this.scatterGraph.getData().add(this.testingPoints);
    }

    @Override
    public void startReport(List<String> metaCols, List<String> labels) {
        // Save the index of the label of interest.
        this.labelIdx = this.cmbLabel.getSelectionModel().getSelectedIndex();
        // Clear the data series.
        this.trainingPoints.getData().clear();
        this.testingPoints.getData().clear();
        // Clear the outlier list.
        this.outliers = new RatingList<Prediction>(MAX_OUTLIERS);
    }

    @Override
    public void reportOutput(List<String> metaData, INDArray expected, INDArray output) {
        // Loop through the metadata, peeling off predictions.
        for (int r = 0; r < metaData.size(); r++) {
            String id = this.getId(metaData.get(r));
            // Create the data point.
            double expect = expected.getDouble(r, this.labelIdx);
            double predict = output.getDouble(r, this.labelIdx);
            XYChart.Data<Double, Double> dataPoint = new XYChart.Data<Double, Double>(expect, predict);
            // Update the outlier list.
            this.processOutlier(id, expect, predict);
            // Determine which series should contain the data.
            if (this.isTrained(id))
                this.trainingPoints.getData().add(dataPoint);
            else
                this.testingPoints.getData().add(dataPoint);
        }
        // Fill in the outlier table.
        this.outlierItems.clear();
        this.outlierItems.addAll(this.outliers.getBest());
    }

    /**
     * Add this data point to the outlier list.  We allow the outlier list to grow past the max if all the
     * predictions at the end have the same error value.
     *
     * @param id		ID of the data point
     * @param expect	expected value
     * @param predict	predicted value
     */
    private void processOutlier(String id, double expect, double predict) {
        Prediction newPoint = new Prediction(id, expect, predict);
        this.outliers.add(newPoint, Math.abs(expect - predict));
    }

    @Override
    public List<Stat> getStats() {
        return Collections.emptyList();
    }

    /**
      * This nested class listens for a row selection in the outlier table.
      */
    public class RowListener implements ChangeListener<Prediction> {

        @Override
        public void changed(ObservableValue<? extends Prediction> observable, Prediction oldValue,
                Prediction newValue) {
            // Save the item ID in the clipboard.
            final Clipboard clipboard = Clipboard.getSystemClipboard();
            ClipboardContent content = new ClipboardContent();
            content.putString(newValue.getId());
            clipboard.setContent(content);
        }
    }

    @Override
    public void registerOutlierTable(TableView<Prediction> outlierTable) {
        // Set up the table columns.
        Prediction.setupTable(outlierTable);
        // Get the table's row list.
        this.outlierItems = outlierTable.getItems();
        // Create the selection event.
        outlierTable.getSelectionModel().selectedItemProperty().addListener(new RowListener());
    }

    @Override
    public void finishReport() {
        ObservableList<XYChart.Data<Double, Double>> testingData = this.testingPoints.getData();
        for (XYChart.Data<Double, Double> testingPoint : testingData)
            testingPoint.getNode().toFront();
    }

}
