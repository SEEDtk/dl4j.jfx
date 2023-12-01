/**
 *
 */
package org.theseed.dl4j.jfx.parms;

import javafx.event.ActionEvent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.theseed.io.ParmDescriptor;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Region;
import javafx.event.EventHandler;

/**
 * This is the base class for a contol group in the parameter file edit dialog.  The control group
 * consists of an enable/disable checkbox, the parameter name (with a tooltip containing the
 * description), and the parameter value editor.  The value editor is controlled by the subclass.
 * There are many parameter types, each with its own subclass, including text parameters,
 * enumerations, and flags.  Most parameters have a syntax allowing the user to select multiple
 * values for searching.
 *
 * @author Bruce Parrello
 *
 */
public abstract class ParmDialogGroup {

    // FIELDS
    /** parameter descriptor being managed */
    private ParmDescriptor descriptor;
    /** linked parameters-- these are enabled and disabled in concert */
    private List<ParmDialogGroup> others;
    /** mutually exclusive parameters-- these are disabled when we are enabled */
    private List<ParmDialogGroup> exclusives;
    /** parent container, a grid pane with width 3 */
    private GridPane parent;

    // CONTROLS

    /** checkbox for enable/disable */
    private CheckBox checkBox;

    /** main control for entering value */
    private Region mainControl;

    /**
     * Set up a parameter dialog group and make it ready for use.
     *
     * @param parent	parent container
     * @param row		row index in parent container for this parameter
     * @param desc		descriptor with specs for the parameter
     */
    public void init(GridPane parent, int row, ParmDescriptor desc) {
        // Initialize the lists.
        this.others = new ArrayList<ParmDialogGroup>();
        this.exclusives = new ArrayList<ParmDialogGroup>();
        // Connect to the container and the descriptor.
        this.parent = parent;
        this.descriptor = desc;
        // Create the checkbox.
        this.checkBox = new CheckBox();
        this.parent.add(this.checkBox, 0, row);
        this.checkBox.setSelected(! desc.isCommented());
        this.checkBox.setOnAction(this.new CheckListener());
        // Label the parameter.
        Label label = new Label(desc.getName());
        this.parent.add(label, 1, row);
        Tooltip comment = new Tooltip(desc.getDescription());
        Tooltip.install(label, comment);
        // Create the main control/
        this.mainControl = this.createMainControl();
        this.parent.add(this.mainControl, 2, row);
        // Perform post-layout processing.
        this.initMainControl();
        // Configure the enable state.
        this.configure();
    }

    /**
     * Configure the enable state of the main control based on the checkbox state.
     * If we are commented out, disable it.
     */
    private void configure() {
        this.mainControl.setDisable(this.descriptor.isCommented());
    }

    /**
     * Create the main control for this parameter.  Note it is not inserted into the grid in
     * this method.
     *
     * @return the main control for this parameter
     */
    protected abstract Region createMainControl();

    /**
     * Perform post-layout initialization of the main control.
     */
    protected abstract void initMainControl();

    /**
     * Update the descriptor and configure the states.  This method is used when handling
     * mutually exclusive and dependent parameters.
     *
     * @param newState	TRUE to enable, FALSE to disable
     */
    private void configure(boolean newState) {
        // Note that commented is disabled, uncommented is enabled.
        this.descriptor.setCommented(! newState);
        // Update the checkbox.
        this.checkBox.setSelected(newState);
        // Configure the controls.
        this.configure();
    }

    /**
     * Specify mutually exclusive parameters.
     *
     * @param groups	array of groups to mark mutually exclusive
     */
    public void setExclusive(ParmDialogGroup... groups) {
        Arrays.stream(groups).filter(x -> x != null).forEach(x -> this.exclusives.add(x));
    }

    /**
     * Specify grouped parameters.  Only one parameter should do this.  The
     * others will have their checkboxes disabled.
     *
     * @param groups	array of groups to mark as subordinate to this one
     */
    public void setGrouped(ParmDialogGroup... groups) {
        for (ParmDialogGroup group : groups) {
            if (group != null) {
                this.others.add(group);
                group.checkBox.setDisable(true);
            }
        }
    }

    /**
     * Inner class for handling checkbox events.
     */
    private class CheckListener implements  EventHandler<ActionEvent> {

        @Override
        public void handle(ActionEvent event) {
            descriptor.setCommented(! checkBox.isSelected());
            configure();
            for (ParmDialogGroup other : others)
                other.configure(! descriptor.isCommented());
            for (ParmDialogGroup exclusive : exclusives)
                exclusive.configure(descriptor.isCommented());
        }
    }

    /**
     * Update the parameter value.
     *
     * @param	newValue		new parameter value
     */
    protected void setValue(String newValue) {
        this.descriptor.setValue(newValue);
    }

    /**
     * @return the descriptor
     */
    protected ParmDescriptor getDescriptor() {
        return this.descriptor;
    }

    /**
     * @return the parent
     */
    protected GridPane getParent() {
        return this.parent;
    }

    /**
     * @return the parameter description
     */
    protected String getDescription() {
        return this.descriptor.getDescription();
    }

}
