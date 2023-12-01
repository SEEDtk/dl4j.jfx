/**
 *
 */
package org.theseed.join;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.IntStream;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.io.KeyedFileMap;
import org.theseed.join.ColumnData.Type;

import j2html.tags.ContainerTag;
import j2html.tags.DomContent;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.CheckBox;
import javafx.scene.control.TextField;
import javafx.stage.FileChooser.ExtensionFilter;

import static j2html.TagCreator.*;


/**
 * This operation saves the file in progress to a web page.  Web pages do not have the
 * power of Excel, but they allow pubmed linking, and some files that will not fit in Excel
 * will fit on a web page.
 *
 * @author Bruce Parrello
 *
 */
public class HtmlSaveSpec extends SaveSpec {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(ExcelSaveSpec.class);
    /** HTML output file extension filter */
    private static final ExtensionFilter HTML_FILTER = new ExtensionFilter("HTML file",
            "*.html", "*.htm");
    /** HTML tag for styles */
    private static final ContainerTag STYLES = style("td.num, th.num { text-align: right; }\n" +
            "td.flag, th.flag { text-align: center; }\ntd.text, th.text { text-align: left; }\n" +
            "td, th { border-style: groove; padding: 2px; vertical-align: top; }\n" +
            "table { border-collapse: collapse; width: 95vw }");

    // CONTROLS

    /** checkbox for pubmed linking */
    @FXML
    private CheckBox chkPubmed;

    /** column name for pubmed linking */
    @FXML
    private TextField txtPubmed;

    /** text box for page title */
    @FXML
    private TextField txtPageTitle;

    @Override
    public boolean isValid() {
        return (this.getOutFile() != null);
    }

    @Override
    public void apply(KeyedFileMap keyedMap) throws IOException {
        // The tricky part of this is figuring out which columns are integers, which
        // gets pubmed-linked, and which are floating-point numbers.  As always, the
        // key column is always text.
        List<String> headers = keyedMap.getHeaders();
        int nCols = headers.size();
        Collection<List<String>> records = keyedMap.getRecords();
        ColumnData.Type[] types = this.computeColumnTypes(nCols, records);
        // Find the pubmed column.
        int pubmedCol;
        if (! chkPubmed.isSelected())
            pubmedCol = -1;
        else {
            pubmedCol = keyedMap.findColumn(txtPubmed.getText());
            if (pubmedCol < 0)
                throw new IOException("Pubmed column not found.");
        }
        // Build the table header row.
        ContainerTag headerRow = tr();
        IntStream.range(0, nCols)
                .forEach(i -> headerRow.with(this.setClass(th(headers.get(i)), types[i])));
        ContainerTag table = table().with(headerRow);
        // Build the data rows.
        for (List<String> record : records) {
            ContainerTag dataRow = tr();
            for (int c = 0; c < nCols; c++) {
                String text = record.get(c);
                if (StringUtils.isBlank(text)) {
                    // In a blank cell, we need to show a non-breaking space.
                    dataRow.with(td(rawHtml("&nbsp;")));
                } else {
                    DomContent cellContent;
                    if (c == pubmedCol) {
                        // Here we have a pubmed link.
                        String url = String.format(ExcelSaveSpec.PUBMED_FORMAT, text);
                        cellContent = a(text).withHref(url);
                    } else {
                        // Here we have a normal cell.
                        cellContent = text(text);
                    }
                    // Put the cell in the data row.
                    dataRow.with(this.setClass(td(cellContent), types[c]));
                }
            }
            // Put the data row in the table.
            table.with(dataRow);
        }
        // Format the output page.
        String title = this.txtPageTitle.getText();
        ContainerTag page = html(head(title(title), STYLES), body(h1(title), table));
        // Write it to the output.
        try (PrintWriter writer = new PrintWriter(this.getOutFile())) {
            writer.println(page.render());
        }
        // Open the file in the browser if requested.
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
     * Compute the column type of each column in a set of records, each having a fixed
     * number of columns.
     *
     * @param nCols		number of columns in each record
     * @param records	records to be examined
     *
     * @return an array containing the type of each column
     */
    public ColumnData.Type[] computeColumnTypes(int nCols, Collection<List<String>> records) {
        ColumnData[] colSpecs = IntStream.range(0, nCols).mapToObj(i -> new ColumnData())
                .toArray(ColumnData[]::new);
        // The key column is always text.  We don't scan it, and set the type manually at the
        // end.
        for (List<String> record : records)
            IntStream.range(1,  nCols).forEach(i -> colSpecs[i].check(record.get(i)));
        ColumnData.Type[] retVal = Arrays.stream(colSpecs).map(x -> x.getType())
                .toArray(Type[]::new);
        retVal[0] = ColumnData.Type.TEXT;
        return retVal;
    }

    /**
     * Specify the style class for this table cell.
     *
     * @param cell		table cell
     * @param type		column data type
     *
     * @return the table cell, for fluent construction
     */
    private DomContent setClass(ContainerTag cell, ColumnData.Type type) {
        switch (type) {
        case INTEGER :
        case DOUBLE :
            cell.withClass("num");
            break;
        case FLAG :
            cell.withClass("flag");
            break;
        default :
            cell.withClass("text");
        }
        return cell;
    }

    @Override
    protected void initControls() {
        // Initialize the page title
        this.txtPageTitle.setText("Combined Output");
    }

    @Override
    protected String getFileLabel() {
        return "Web File";
    }

    @Override
    protected ExtensionFilter getFilter() {
        return HTML_FILTER;
    }

}
