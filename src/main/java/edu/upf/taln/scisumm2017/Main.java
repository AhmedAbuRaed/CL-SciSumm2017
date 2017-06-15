package edu.upf.taln.scisumm2017;

import edu.upf.taln.dri.lib.Factory;
import edu.upf.taln.dri.lib.exception.DRIexception;
import edu.upf.taln.scisumm2017.preprocess.PreProcessPipeline;

/**
 * Created by Ahmed on 6/7/17.
 */
public class Main {
    public static String workingDir;
    public static void main(String args[]) {
        workingDir = args[2].trim();

        System.out.println("Initializing Dr. Inventor Framework ...");
        try {
            // A) IMPORTANT: Initialize Dr. Inventor Framework
            // A.1) set the local path to the config_file previously downloaded
            Factory.setDRIPropertyFilePath(workingDir + "/DRIconfig.properties");
            // A.2) Initialize the Dr. Inventor Text Mining Framework
            Factory.initFramework();
        } catch (DRIexception drIexception) {
            drIexception.printStackTrace();
        }

        switch (args[1])
        {
            case "PreProcessPipeline":
                PreProcessPipeline.PreProcess(args);
                break;
        }
    }
}
