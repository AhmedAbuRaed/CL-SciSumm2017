package edu.upf.taln.scisumm2017.preprocess;

import edu.upf.taln.scisumm2017.Main;
import edu.upf.taln.scisumm2017.Utilities;
import gate.AnnotationSet;
import gate.Document;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.*;
import java.util.HashMap;

/**
 * Created by Ahmed on 6/7/17.
 */
public class PreProcessPipeline {
    public static void PreProcess(String workingDirectory, String datasetType, String target, boolean isTrain,
                                  Word2Vec gvec, Word2Vec aclvec, String [] components) {

        if (target.equals("ALL")) {
            File corpus = new File(workingDirectory + "/datasets/" + datasetType);

            for (File folder : corpus.listFiles()) {
                System.out.println("PreProcessing Cluster:" + folder.getName());
                File inputFolder = new File(folder.getPath() + File.separator + folder.getName());
                HashMap<String, Document> documents = Utilities.extractDocumentsFromBaseFolder(inputFolder);

                for (int i = 0; i < components.length; i++) {
                    if (components[i].equals("GS")) {
                        documents = GoldAnnotations.run(documents, folder, isTrain);
                    }
                    if (components[i].equals("DI")) {
                        PrintStream out = System.out;
                        System.setOut(new PrintStream(new OutputStream() {
                            @Override
                            public void write(int b) throws IOException {
                            }
                        }));
                        try
                        {
                            documents = DRInventor.run(documents);
                        } catch (Exception e) {
                            e.printStackTrace();
                        } finally {
                            System.setOut(out);
                        }
                        System.gc();
                    }
                    if(components[i].equals("BN"))
                    {
                        documents = Babelfy.run(documents);
                        System.gc();
                    }
                    if (components[i].equals("CV")) {
                        for(String key: documents.keySet())
                        {
                            documents.put(key, Utilities.fillDocumentMissingLemmas(documents.get(key)));
                            documents.put(key, Utilities.fillDocumentBabelNetKind(documents.get(key)));
                        }

                        documents = ContextVectors.run(documents, workingDirectory);
                        System.gc();
                    }
                    if(components[i].equals("GZ"))
                    {
                        for(String key: documents.keySet())
                        {
                            documents.put(key, Utilities.fillDocumentMissingPOS(documents.get(key)));
                        }
                        documents = Gazetteers.run(documents, workingDirectory);
                        System.gc();
                    }
                    if(components[i].equals("NG"))
                    {
                        documents = NGrams.run(documents, workingDirectory);
                        System.gc();
                    }
                    if (components[i].equals("WE")) {
                        documents = WordEmbedding.run(documents, gvec, "GoogleNews");
                        System.gc();

                        documents = WordEmbedding.run(documents, aclvec, "ACL");
                        System.gc();
                    }
                }
                Utilities.exportGATEDocuments(documents, folder.getName(), folder.getPath() + "/output", "PreProcessed");
            }

        } else if (!target.equals("ALL")) {
            File clusterFolder = new File(workingDirectory + "/datasets/" + datasetType + File.separator + target);
            System.out.println("PreProcessing Cluster:" + clusterFolder.getName());
            File inputFolder = new File(clusterFolder.getPath() + File.separator + clusterFolder.getName());
            HashMap<String, Document> documents = Utilities.extractDocumentsFromBaseFolder(inputFolder);

            for (int i = 4; i < components.length; i++) {
                if (components[i].equals("GS")) {
                    documents = GoldAnnotations.run(documents, clusterFolder, isTrain);
                    System.gc();
                }
                if (components[i].equals("DI")) {
                    PrintStream out = System.out;
                    System.setOut(new PrintStream(new OutputStream() {
                        @Override
                        public void write(int b) throws IOException {
                        }
                    }));
                    try
                    {
                        documents = DRInventor.run(documents);
                    } catch (Exception e) {
                        e.printStackTrace();
                    } finally {
                        System.setOut(out);
                    }
                    System.gc();
                }
                if(components[i].equals("BN"))
                {
                    documents = Babelfy.run(documents);
                    System.gc();
                }
                if (components[i].equals("CV")) {
                    for(String key: documents.keySet())
                    {
                        documents.put(key, Utilities.fillDocumentMissingLemmas(documents.get(key)));
                        documents.put(key, Utilities.fillDocumentBabelNetKind(documents.get(key)));
                    }

                    documents = ContextVectors.run(documents, workingDirectory);
                    System.gc();
                }
                if(components[i].equals("GZ"))
                {
                    for(String key: documents.keySet())
                    {
                        documents.put(key, Utilities.fillDocumentMissingPOS(documents.get(key)));
                    }
                    documents = Gazetteers.run(documents, workingDirectory);
                    System.gc();
                }
                if(components[i].equals("NG"))
                {
                    documents = NGrams.run(documents, workingDirectory);
                    System.gc();
                }
                if (components[i].equals("WE")) {
                    documents = WordEmbedding.run(documents, gvec, "GoogleNews");

                    gvec = null;
                    System.gc();

                    documents = WordEmbedding.run(documents, aclvec, "ACL");

                    aclvec = null;
                    System.gc();
                }
            }
            Utilities.exportGATEDocuments(documents, clusterFolder.getName(), clusterFolder.getPath() + "/output", "PreProcessed");
        }
    }
}
