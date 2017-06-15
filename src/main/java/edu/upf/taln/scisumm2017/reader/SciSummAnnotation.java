package edu.upf.taln.scisumm2017.reader;

import java.util.ArrayList;

/**
 * Created by Ahmed on 5/5/17.
 */
public class SciSummAnnotation {
    private String citance_Number;
    private String reference_Article;
    private String citing_Article;
    private String citation_Marker_Offset;
    private String citation_Marker;
    private ArrayList<String> citation_Offset = new ArrayList<String>();
    private String citation_Text;
    private ArrayList<String> reference_Offset = new ArrayList<String>();
    private String reference_Text;
    private ArrayList<String> discourse_Facet = new ArrayList<String>();
    private String annotator;

    public String getCitance_Number() {
        return citance_Number;
    }

    public void setCitance_Number(String citance_Number) {
        this.citance_Number = citance_Number;
    }

    public String getReference_Article() {
        return reference_Article;
    }

    public void setReference_Article(String reference_Article) {
        this.reference_Article = reference_Article;
    }

    public String getCiting_Article() {
        return citing_Article;
    }

    public void setCiting_Article(String citing_Article) {
        this.citing_Article = citing_Article;
    }

    public String getCitation_Marker_Offset() {
        return citation_Marker_Offset;
    }

    public void setCitation_Marker_Offset(String citation_Marker_Offset) {
        this.citation_Marker_Offset = citation_Marker_Offset;
    }

    public String getCitation_Marker() {
        return citation_Marker;
    }

    public void setCitation_Marker(String citation_Marker) {
        this.citation_Marker = citation_Marker;
    }

    public ArrayList<String> getCitation_Offset() {
        return citation_Offset;
    }

    public void setCitation_Offset(ArrayList<String> citation_Offset) {
        this.citation_Offset = citation_Offset;
    }

    public String getCitation_Text() {
        return citation_Text;
    }

    public void setCitation_Text(String citation_Text) {
        this.citation_Text = citation_Text;
    }

    public ArrayList<String> getReference_Offset() {
        return reference_Offset;
    }

    public void setReference_Offset(ArrayList<String> reference_Offset) {
        this.reference_Offset = reference_Offset;
    }

    public String getReference_Text() {
        return reference_Text;
    }

    public void setReference_Text(String reference_Text) {
        this.reference_Text = reference_Text;
    }

    public ArrayList<String> getDiscourse_Facet() {
        return discourse_Facet;
    }

    public void setDiscourse_Facet(ArrayList<String> discourse_Facet) {
        this.discourse_Facet = discourse_Facet;
    }

    public String getAnnotator() {
        return annotator;
    }

    public void setAnnotator(String annotator) {
        this.annotator = annotator;
    }
}