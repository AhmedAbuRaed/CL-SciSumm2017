package edu.upf.taln.scisumm2017.preprocess;

import edu.upf.taln.scisumm2017.Main;
import edu.upf.taln.scisumm2017.Utilities;
import gate.Document;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.HashMap;

/**
 * Created by Ahmed on 6/7/17.
 */
public class PreProcessPipeline {
    public static void PreProcess(String args[]) {
        String workingDirectory = args[2];

        String[] targetOptions = args[3].split("\\_");
        String target = targetOptions[0];
        String datasetType = targetOptions[1];
        boolean isTrain;
        if (targetOptions[2].equals("train")) {
            isTrain = true;
        } else {
            isTrain = false;
        }

        if (target.equals("ALL")) {
            File corpus = new File(workingDirectory + "/datasets/" + datasetType);
            for (File folder : corpus.listFiles()) {
                System.out.println("PreProcessing Cluster:" + folder.getName());
                File inputFolder = new File(folder.getPath() + File.separator + folder.getName());
                HashMap<String, Document> documents = Utilities.extractDocumentsFromBaseFolder(inputFolder);

                for (int i = 4; i < args.length; i++) {
                    if (args[i].equals("GS")) {
                        documents = GoldAnnotations.run(documents, folder, isTrain);
                    }
                    if (args[i].equals("DI")) {
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
                    }
                    if(args[i].equals("BN"))
                    {
                        documents = Babelfy.run(documents);
                    }
                    if (args[i].equals("CV")) {

                    }
                    if (args[i].equals("WE")) {
                        //Get file from resources folder
                        ClassLoader classLoader = Main.class.getClassLoader();
                        File gModel = new File(classLoader.getResource("GoogleNews-vectors-negative300.bin.gz").getFile());
                        Word2Vec gvec = WordVectorSerializer.readWord2VecModel(gModel);

                        File aclModel = new File(classLoader.getResource("ACL300").getFile());
                        Word2Vec aclvec = WordVectorSerializer.readWord2VecModel(aclModel);
                    }
                }
                Utilities.exportGATEDocuments(documents, folder.getName(), folder.getPath() + "/output", "PreProcessed");
            }

        } else if (!target.equals("ALL")) {
            File clusterFolder = new File(workingDirectory + "/datasets/" + datasetType + File.separator + target);
            System.out.println("PreProcessing Cluster:" + clusterFolder.getName());
            File inputFolder = new File(clusterFolder.getPath() + File.separator + clusterFolder.getName());
            HashMap<String, Document> documents = Utilities.extractDocumentsFromBaseFolder(inputFolder);

            for (int i = 4; i < args.length; i++) {
                if (args[i].equals("GS")) {
                    documents = GoldAnnotations.run(documents, clusterFolder, isTrain);
                }
                if (args[i].equals("DI")) {
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
                }
                if(args[i].equals("BN"))
                {
                    documents = Babelfy.run(documents);
                }
                if (args[i].equals("CV")) {

                }
                if (args[i].equals("WE")) {
                    //Get file from resources folder
                    ClassLoader classLoader = Main.class.getClassLoader();
                    File gModel = new File(classLoader.getResource("GoogleNews-vectors-negative300.bin.gz").getFile());
                    Word2Vec gvec = WordVectorSerializer.readWord2VecModel(gModel);

                    File aclModel = new File(classLoader.getResource("ACL300").getFile());
                    Word2Vec aclvec = WordVectorSerializer.readWord2VecModel(aclModel);
                }
            }
            Utilities.exportGATEDocuments(documents, clusterFolder.getName(), clusterFolder.getPath() + "/output", "GOLD");
        }
    }
}
