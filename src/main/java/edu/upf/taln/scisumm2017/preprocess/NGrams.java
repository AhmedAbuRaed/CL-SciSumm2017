package edu.upf.taln.scisumm2017.preprocess;

import gate.Corpus;
import gate.CorpusController;
import gate.Document;
import gate.Factory;
import gate.creole.ResourceInstantiationException;
import gate.persist.PersistenceException;
import gate.util.persistence.PersistenceManager;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

/**
 * Created by ahmed on 6/25/2017.
 */
public class NGrams {
    public static CorpusController application;
    public static HashMap<String, Document> run(HashMap<String, Document> rawDocuments, String workingDirectory) {
        try {
            // load the GAPP
            application = (CorpusController) PersistenceManager.loadObjectFromFile(new File(workingDirectory + File.separator + "nGramsCorpusPipelineApp.gapp"));

            Corpus corpus = null;
            for (String key : rawDocuments.keySet()) {
                System.out.println("Generating NGrams for Document: " + key);
                corpus = Factory.newCorpus("");
                corpus.add(rawDocuments.get(key));
                application.setCorpus(corpus);
                application.execute();

                rawDocuments.put(key, rawDocuments.get(key));
                Factory.deleteResource(corpus);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (PersistenceException e) {
            e.printStackTrace();
        } catch (ResourceInstantiationException e) {
            e.printStackTrace();
        } catch (gate.creole.ExecutionException e) {
            e.printStackTrace();
        }
        return rawDocuments;
    }
}
