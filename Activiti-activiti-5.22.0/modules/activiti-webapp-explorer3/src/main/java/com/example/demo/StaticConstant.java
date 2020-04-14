package com.example.demo;

import java.io.File;

public class StaticConstant {
    private static File dir = new File(".");
    public static String pythoyScriptSubDirPath = "predictive-maintenance\\";
//    public static String pythoyScriptHomeDirPath = System.getProperty("java.class.path");
    public static String pythoyScriptHomeDirPath = StaticConstant.class.getResource("").getPath().substring(1);

    public static String getPythoyScriptDirPath(){
       return  pythoyScriptHomeDirPath.replace("%20"," ").replace("/","\\").split("activiti-webapp-explorer3")[0] + pythoyScriptSubDirPath;
    }
}
