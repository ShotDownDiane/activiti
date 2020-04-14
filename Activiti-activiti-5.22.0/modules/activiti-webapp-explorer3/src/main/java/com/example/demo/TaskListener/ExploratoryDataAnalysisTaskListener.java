package com.example.demo.TaskListener;

import com.example.demo.StaticConstant;
import org.activiti.engine.delegate.DelegateTask;
import org.activiti.engine.delegate.TaskListener;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class ExploratoryDataAnalysisTaskListener implements TaskListener {

    @Override
    public void notify(DelegateTask delegateTask) {
        try {
            System.out.println("Task [Exploratory Data Analysis] begins ...");
            String filename = "Exploratory Data Analysis.py";
            Process proc = Runtime.getRuntime().exec("python \""+ StaticConstant.getPythoyScriptDirPath()+filename+"\"");
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = null;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            proc.waitFor();
            System.out.println("Task [Exploratory Data Analysis] finish");
        } catch (IOException e) {
            e.printStackTrace();
        }
        catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
