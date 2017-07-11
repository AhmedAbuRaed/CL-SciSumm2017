package edu.upf.taln.scisumm2017.feature.context;

import gate.Document;

/**
 * Created by ahmed on 5/7/2016.
 */
public class DocumentCtx {
    private Document rp, cp;

    public DocumentCtx(Document rpDocument, Document cpDocument) {
        this.rp = rpDocument;
        this.cp = cpDocument;

    }

    public Document getReferenceDoc() {
        return rp;
    }

    public Document getCitationDoc() {
        return cp;
    }
}
