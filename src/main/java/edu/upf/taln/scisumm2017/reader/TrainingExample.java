package edu.upf.taln.scisumm2017.reader;

import gate.Annotation;

/**
 * Created by ahmed on 5/8/2016.
 */
public class TrainingExample {

    private Annotation referenceSentence;
    private Annotation citanceSentence;
    private String facet;
    private Integer isMatch;

    public TrainingExample(Annotation referenceSentence, Annotation citanceSentence, String facet) {
        this.setReferenceSentence(referenceSentence);
        this.setCitanceSentence(citanceSentence);
        this.setFacet(facet);
    }

    public TrainingExample(Annotation referenceSentence, Annotation citanceSentence, Integer isMatch) {
        this.setReferenceSentence(referenceSentence);
        this.setCitanceSentence(citanceSentence);
        this.setIsMatch(isMatch);
    }

    public Annotation getReferenceSentence() {
        return referenceSentence;
    }

    public void setReferenceSentence(Annotation referenceSentence) {
        this.referenceSentence = referenceSentence;
    }

    public Annotation getCitanceSentence() {
        return citanceSentence;
    }

    public void setCitanceSentence(Annotation citanceSentence) {
        this.citanceSentence = citanceSentence;
    }

    public String getFacet() {
        return facet;
    }

    public void setFacet(String facet) {
        this.facet = facet;
    }

    public Integer getIsMatch() {
        return isMatch;
    }

    public void setIsMatch(Integer isMatch) {
        this.isMatch = isMatch;
    }
}
