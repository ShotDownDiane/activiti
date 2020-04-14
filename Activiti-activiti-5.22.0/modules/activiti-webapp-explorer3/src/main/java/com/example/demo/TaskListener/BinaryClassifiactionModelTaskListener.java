package com.example.demo.TaskListener;

import com.example.demo.StaticConstant;
import org.activiti.engine.delegate.DelegateTask;
import org.activiti.engine.delegate.TaskListener;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class BinaryClassifiactionModelTaskListener implements TaskListener {

    @Override
    public void notify(DelegateTask delegateTask) {
        String filename = "Model Selection - Binary Classifiaction.py";
        try {
            System.out.println("Task [Binary Classifiaction Modeling] begins ...");
            Process proc = Runtime.getRuntime().exec("python \""+ StaticConstant.getPythoyScriptDirPath()+filename+"\"");
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = null;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            proc.waitFor();
            System.out.println("Task [Binary Classifiaction Modeling] finish.");
        } catch (IOException e) {
            e.printStackTrace();
        }
        catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
