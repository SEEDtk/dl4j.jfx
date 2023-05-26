/**
 *
 */
package org.theseed.dl4j.jfx;

import java.io.File;
import java.io.IOException;

import org.nd4j.linalg.activations.Activation;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.weights.WeightInit;
import org.theseed.dl4j.LossFunctionType;
import org.theseed.dl4j.Regularization;
import org.theseed.dl4j.decision.RandomForest;
import org.theseed.dl4j.decision.TreeFeatureSelectorFactory;
import org.theseed.dl4j.jfx.parms.ParmDialogGroup;
import org.theseed.dl4j.jfx.parms.ParmPaneBuilder;
import org.theseed.dl4j.train.ClassMetric;
import org.theseed.dl4j.train.GradientUpdater;
import org.theseed.dl4j.train.ModelType;
import org.theseed.dl4j.train.Trainer;
import org.theseed.io.ParmFile;
import org.theseed.jfx.BaseController;
import org.theseed.jfx.MovableController;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.layout.GridPane;

/**
 * @author Bruce Parrello
 *
 */
public class ParmDialog extends MovableController {

    // FIELDS
    /** TRUE if the user hit "SAVE AND RUN", FALSE for "CANCEL" */
    private boolean okFlag;
    /** processor model type */
    private ModelType modelType;
    /** parameter map */
    private ParmFile parms;
    /** parameter file name */
    private File parmFile;

    // CONTROLS

    /** grid pane for structure parameters */
    @FXML
    private GridPane structurePane;

    /** grid pane for tuning parameters */
    @FXML
    private GridPane tuningPane;

    /**
     * Create the dialog box.
     */
    public ParmDialog() {
        super(200, 200);
        this.okFlag = false;
    }

    @Override
    public String getIconName() {
        return "multi-gear-16.png";
    }

    @Override
    public String getWindowTitle() {
        return "Set Parameters for Training Search";
    }

    /**
     * Initialize this dialog.
     *
     * @param parmFile		parameter file to read and modify
     * @param modelType		type of model-- CLASS or REGRESSION
     */
    public void init(File parmFile, ModelType modelType) {
        // Save the model type and parm file name.
        this.parmFile = parmFile;
        this.modelType = modelType;
        // Now we read in the parameter file and build the parameters.
        try {
            this.parms = new ParmFile(parmFile);
        } catch (IOException e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Reading Parm File", e.toString());
        }
        // Now we need to build the parameter dialog groups.  First the model structure parameters.
        ParmPaneBuilder builder = new ParmPaneBuilder(this.structurePane, this.parms);
        if (modelType == ModelType.DECISION)
            deciderStructurePane(builder);
        else
            neuralStructurePane(builder);
        // Next, the search tuning parameters.
        builder = new ParmPaneBuilder(this.tuningPane, this.parms);
        if (modelType == ModelType.DECISION)
            deciderSearchPane(builder);
        else
            neuralSearchPane(builder);
    }

    /**
     * Create the parameters for the search pane in a decision model.
     *
     * @param builder	builder for the search pane
     */
    private void deciderSearchPane(ParmPaneBuilder builder) {
        builder.addChoices("prefer", ClassMetric.values());
        builder.addText("meta");
        builder.addText("col");
        builder.addText("id");
        builder.addText("seed");
        builder.addChoices("method", RandomForest.Method.values());
        builder.addText("testSize");
        builder.addText("batchSize");
    }

    /**
     * Create the parameters for the structure pane in a decision model.
     *
     * @param builder	builder for the structure pane
     */
    private void deciderStructurePane(ParmPaneBuilder builder) {
        builder.addText("maxFeatures");
        builder.addText("nEstimators");
        builder.addText("minSplit");
        builder.addText("maxDepth");
        builder.addText("sampleSize");
        builder.addChoices("selection", TreeFeatureSelectorFactory.Type.values());
        builder.addText("rootFile");
    }

    /**
     * Create the parameters for the search pane in a neural net model.
     *
     * @param builder	builder for the search pane
     */
    public void neuralSearchPane(ParmPaneBuilder builder) {
        Enum<?>[] preferTypes = this.modelType.getPreferTypes();
        builder.addChoices("prefer", preferTypes);
        builder.addChoices("method", Trainer.Type.values());
        builder.addText("bound");
        builder.addChoices("lossFun", LossFunctionType.values());
        builder.addText("weights");
        builder.addText("iter");
        builder.addText("batchSize");
        builder.addText("testSize");
        builder.addText("maxBatches");
        builder.addText("earlyStop");
        builder.addChoices("regMode", Regularization.Mode.values());
        builder.addText("regFactor");
        builder.addText("seed");
        builder.addChoices("start", WeightInit.values());
        builder.addChoices("gradNorm", GradientNormalization.values());
        builder.addChoices("updater", GradientUpdater.Type.values());
        builder.addText("learnRate");
        builder.addChoices("bUpdater", GradientUpdater.Type.values());
        builder.addText("updateRate");
    }

    /**
     * Create the parameters for the structure pane in a neural net model.
     *
     * @param builder	builder for the structure pane
     */
    public void neuralStructurePane(ParmPaneBuilder builder) {
        builder.addText("meta");
        builder.addText("col");
        builder.addText("id");
        builder.addChoices("init", Activation.values());
        builder.addChoices("activation", Activation.values());
        builder.addFlag("raw");
        builder.addFlag("batch");
        ParmDialogGroup cnn = builder.addText("cnn");
        ParmDialogGroup filters = builder.addText("filters");
        ParmDialogGroup strides = builder.addText("strides");
        ParmDialogGroup subs = builder.addText("sub");
        if (cnn != null)
            cnn.setGrouped(filters, strides, subs);
        builder.addText("lstm");
        ParmDialogGroup wGroup = builder.addText("widths");
        ParmDialogGroup bGroup = builder.addText("balanced");
        if (bGroup != null && wGroup != null) {
            wGroup.setExclusive(bGroup);
            bGroup.setExclusive(wGroup);
        }
    }

    /**
     * @return TRUE if successful, FALSE if the user cancelled out
     */
    public boolean getResult() {
        return this.okFlag;
    }

    /**
     * Button event to save the parameter file without exiting.
     *
     * @param event		event descriptor
     */
    @FXML
    private void saveAndStay(ActionEvent event) {
        if (this.save())
            BaseController.messageBox(Alert.AlertType.INFORMATION, "Parm File Status", "Parameters saved to " + this.parmFile.getAbsolutePath() + ".");
    }

    /**
     * Button event to exit without saving.
     *
     * @param event		event descriptor
     */
    @FXML
    private void exitNoSave() {
        this.okFlag = false;
        this.close();
    }

    /**
     * Button to save and exit successfully.
     *
     * @param event		event descriptor
     */
    @FXML
    private void saveAndRun() {
        if (this.save()) {
            this.okFlag = true;
            this.close();
        }
    }

    /**
     * Save the parameter file.
     *
     * @return TRUE if successful, else FALSE
     */
    public boolean save() {
        boolean retVal = false;
        try {
            this.parms.save(this.parmFile);
            retVal = true;
        } catch (IOException e) {
            BaseController.messageBox(Alert.AlertType.ERROR, "Error Saving Parm File", e.toString());
        }
        return retVal;
    }

}
