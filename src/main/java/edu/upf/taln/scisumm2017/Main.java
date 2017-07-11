package edu.upf.taln.scisumm2017;

import edu.upf.taln.dri.lib.Factory;
import edu.upf.taln.dri.lib.exception.DRIexception;
import edu.upf.taln.scisumm2017.preprocess.PreProcessPipeline;
import edu.upf.taln.scisumm2017.process.ProcessAsTestingPipeline;
import edu.upf.taln.scisumm2017.process.ProcessAsTrainingPipeline;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.File;
import java.util.Arrays;

/**
 * Created by Ahmed on 6/7/17.
 */
public class Main {
    public static void main(String args[]) {

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

        System.out.println("Initializing Dr. Inventor Framework ...");
        try {
            // A) IMPORTANT: Initialize Dr. Inventor Framework
            // A.1) set the local path to the config_file previously downloaded
            Factory.setDRIPropertyFilePath(workingDirectory + "/DRIconfig.properties");
            // A.2) Initialize the Dr. Inventor Text Mining Framework
            Factory.initFramework();
        } catch (DRIexception drIexception) {
            drIexception.printStackTrace();
        }

        Word2Vec gvec = null;
        Word2Vec aclvec = null;

        //Load Word2Vec models in case it is in the pipeline
        for (String component : Arrays.copyOfRange(args, 4, args.length)) {
            if (component.equals("WE")) {
                System.out.println("Loading Word2vec Models ...");
                //Get file from resources folder
                File gModel = new File(workingDirectory + File.separator + "GoogleNews-vectors-negative300.bin.gz");
                gvec = WordVectorSerializer.readWord2VecModel(gModel);

                File aclModel = new File(workingDirectory + File.separator + "ACL300.txt");
                aclvec = WordVectorSerializer.readWord2VecModel(aclModel);
                System.out.println("Word2vec Models Loaded ...");
            }
        }
        //finished loading models

        switch (args[1]) {
            case "PreProcessPipeline":
                PreProcessPipeline.PreProcess(workingDirectory, datasetType, target, isTrain, gvec, aclvec,
                        Arrays.copyOfRange(args, 4, args.length));
                break;
            case "ProcessPipeline":
                if (isTrain) {
                    ProcessAsTrainingPipeline.ProcessAsTraining(workingDirectory, datasetType, target);
                } else {
                    ProcessAsTestingPipeline.ProcessAsTesting();
                }

                break;
        }
    }
}
