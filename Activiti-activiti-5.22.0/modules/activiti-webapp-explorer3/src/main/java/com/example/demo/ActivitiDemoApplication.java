package com.example.demo;

import com.example.demo.Bean.AppayBillBean;
import org.activiti.engine.*;
import org.activiti.engine.identity.User;
import org.activiti.engine.repository.Deployment;
import org.activiti.engine.runtime.ProcessInstance;
import org.activiti.engine.task.Attachment;
import org.activiti.engine.task.Task;
import org.activiti.engine.task.TaskQuery;
import java.io.File;
import java.io.InputStream;
import java.util.Date;
import java.util.List;
import java.util.Scanner;


public class ActivitiDemoApplication {

	private ProcessEngine processEngine = null;
    private ProcessInstance pi = null;

	public static void main(String[] args) {
        ActivitiDemoApplication activitiDemoApplication = new ActivitiDemoApplication();
        activitiDemoApplication.createActitiEngine();
        activitiDemoApplication.deploy();

//        activitiDemoApplication.startProcess();
//        activitiDemoApplication.queryTask();
//        activitiDemoApplication.queryAttachment();
//        int i;
//        for (i=0;i<4;i++){
//            activitiDemoApplication.comlileAllTask();
//        }
//        activitiDemoApplication.processEngine.getRuntimeService().deleteProcessInstance(procInsId, "xxx原因");
//        activitiDemoApplication.setVariable();
//        activitiDemoApplication.getVariable();

    }


//	private static void test (ProcessEngine processEngine,ProcessDefinition processDefinition){
//		RuntimeService runtimeService = processEngine.getRuntimeService();
//		ProcessInstance processInstance = runtimeService
//				.startProcessInstanceByKey("test4");
//		System.out.println(" process started with process instance id ["
//				+ processInstance.getProcessInstanceId()
//				+ "] key [" + processInstance.getProcessDefinitionKey() + "]");
//
//		TaskService taskService = processEngine.getTaskService();
//		FormService formService = processEngine.getFormService();
//		HistoryService historyService = processEngine.getHistoryService();
//
//		Scanner scanner = new Scanner(System.in);
//		while (processInstance != null && !processInstance.isEnded()) {
//			List<Task> tasks = taskService.createTaskQuery()
//					.taskCandidateGroup("managers").list();
//			System.out.println("Active outstanding tasks: [" + tasks.size() + "]");
//			for (int i = 0; i < tasks.size(); i++) {
//				Task task = tasks.get(i);
//				System.out.println("Processing Task [" + task.getName() + "]");
//				Map<String, Object> variables = new HashMap<String, Object>();
//				FormData formData = formService.getTaskFormData(task.getId());
//				for (FormProperty formProperty : formData.getFormProperties()) {
//					if (StringFormType.class.isInstance(formProperty.getType())) {
//						System.out.println(formProperty.getName() + "?");
//						String value = scanner.nextLine();
//						variables.put(formProperty.getId(), value);
//					} else if (LongFormType.class.isInstance(formProperty.getType())) {
//						System.out.println(formProperty.getName() + "? (Must be a whole number)");
//						Long value = Long.valueOf(scanner.nextLine());
//						variables.put(formProperty.getId(), value);
//					} else if (DateFormType.class.isInstance(formProperty.getType())) {
//						System.out.println(formProperty.getName() + "? (Must be a date m/d/yy)");
//						DateFormat dateFormat = new SimpleDateFormat("m/d/yy");
//						Date value = null;
//						try {
//							value = dateFormat.parse(scanner.nextLine());
//						} catch (ParseException e) {
//							e.printStackTrace();
//						}
//						variables.put(formProperty.getId(), value);
//					} else {
//						System.out.println("<form type not supported>");
//					}
//				}
//				taskService.complete(task.getId(), variables);
//
//				HistoricActivityInstance endActivity = null;
//				List<HistoricActivityInstance> activities =
//						historyService.createHistoricActivityInstanceQuery()
//								.processInstanceId(processInstance.getId()).finished()
//								.orderByHistoricActivityInstanceEndTime().asc()
//								.list();
//				for (HistoricActivityInstance activity : activities) {
//					if (activity.getActivityType() == "startEvent") {
//						System.out.println("BEGIN " + processDefinition.getName()
//								+ " [" + processInstance.getProcessDefinitionKey()
//								+ "] " + activity.getStartTime());
//					}
//					if (activity.getActivityType() == "endEvent") {
//						// Handle edge case where end step happens so fast that the end step
//						// and previous step(s) are sorted the same. So, cache the end step
//						//and display it last to represent the logical sequence.
//						endActivity = activity;
//					} else {
//						System.out.println("-- " + activity.getActivityName()
//								+ " [" + activity.getActivityId() + "] "
//								+ activity.getDurationInMillis() + " ms");
//					}
//				}
//				if (endActivity != null) {
//					System.out.println("-- " + endActivity.getActivityName()
//							+ " [" + endActivity.getActivityId() + "] "
//							+ endActivity.getDurationInMillis() + " ms");
//					System.out.println("COMPLETE " + processDefinition.getName() + " ["
//							+ processInstance.getProcessDefinitionKey() + "] "
//							+ endActivity.getEndTime());
//				}
//			}
//			// Re-query the process instance, making sure the latest state is available
//			processInstance = runtimeService.createProcessInstanceQuery()
//					.processInstanceId(processInstance.getId()).singleResult();
//		}
//		scanner.close();
//
//	}


	public void createActitiEngine(){
		ProcessEngineConfiguration engineConfiguration = ProcessEngineConfiguration.createStandaloneProcessEngineConfiguration();
		engineConfiguration.setJdbcDriver("com.mysql.jdbc.Driver");
		engineConfiguration.setJdbcUrl("jdbc:mysql://localhost:3306/activitiDB?createDatabaseIfNotExist=true"
				+ "&useUnicode=true&characterEncoding=utf8&serverTimezone=Asia/Shanghai");
		engineConfiguration.setJdbcUsername("root");
		engineConfiguration.setJdbcPassword("wjy199042");
		engineConfiguration.setDatabaseSchemaUpdate("true");
		processEngine = engineConfiguration.buildProcessEngine();
		System.out.println("ProcessEngin build successfully with name ["+processEngine.getName()+"] ,version ["+processEngine.VERSION+"]");
	}

	public void deploy() {

		//获取仓库服务 ：管理流程定义
		RepositoryService repositoryService = processEngine.getRepositoryService();
		Deployment deploy = repositoryService.createDeployment()//创建一个部署的构建器
				.addClasspathResource("diagrams/pdm_1.bpmn")//从类路径中添加资源,一次只能添加一个资源
				.name("pdm_1")//设置部署的名称
				.deploy();

		System.out.println("部署的id"+deploy.getId());
		System.out.println("部署的名称"+deploy.getName());
	}

    public void startProcess(){

        //指定执行我们刚才部署的工作流程
        String processDefiKey="pdm_1";
        //取运行时服务
        RuntimeService runtimeService = processEngine.getRuntimeService();
        //取得流程实例
        pi = runtimeService.startProcessInstanceByKey(processDefiKey);//通过流程定义的key 来执行流程

        System.out.println("流程执行对象的id："+pi.getId());//Execution 对象
        System.out.println("流程实例的id："+pi.getProcessInstanceId());//ProcessInstance 对象
        System.out.println("流程定义的id："+pi.getProcessDefinitionId());//默认执行的是最新版本的流程定义
    }

    //查询任务
    public void queryTask(){
        //取得任务服务
        TaskService taskService = processEngine.getTaskService();
        //创建一个任务查询对象
        TaskQuery taskQuery = taskService.createTaskQuery();
        //办理人的任务列表
        List<Task> list = taskQuery.list();
        //遍历任务列表
        if(list!=null&&list.size()>0){
            for(Task task:list){
                System.out.println("任务的办理人："+task.getAssignee());
                System.out.println("任务的id："+task.getId());
                System.out.println("任务的名称："+task.getName());

            }
        }

    }

    public void queryTaskByAssignee(String assignee){
        //任务的办理人

        //取得任务服务
        TaskService taskService = processEngine.getTaskService();
        //创建一个任务查询对象
        TaskQuery taskQuery = taskService.createTaskQuery();
        //办理人的任务列表
        List<Task> list = taskQuery.taskAssignee(assignee)//指定办理人
                .list();
        //遍历任务列表
        if(list!=null&&list.size()>0){
            for(Task task:list){
                System.out.println("任务的办理人："+task.getAssignee());
                System.out.println("任务的id："+task.getId());
                System.out.println("任务的名称："+task.getName());
            }
        }
    }

    //获取流程实例的状态 判断流程是否在运行
    public boolean getProcessInstanceState(String processInstanceId){
        ProcessInstance pi = processEngine.getRuntimeService()
                .createProcessInstanceQuery()
                .processInstanceId(processInstanceId)
                .singleResult();//返回的数据要么是单行，要么是空 ，其他情况报错
        //判断流程实例的状态
        if(pi!=null){
            System.out.println("该流程实例"+processInstanceId+"正在运行...  "+"当前活动的任务:"+pi.getActivityId());
            return true;
        }else{
            System.out.println("当前的流程实例"+processInstanceId+" 已经结束！");
            return false;
        }

    }

    public void compileTaskById(String taskId){
        //taskId：任务id
        processEngine.getTaskService().complete(taskId);
        System.out.println(taskId+":当前任务执行完毕");
    }

    public void comlileAllTask(){
        List<Task> list = processEngine.getTaskService().createTaskQuery().list();
        //遍历任务列表
        if(list!=null&&list.size()>0){
            for(Task task:list){
                processEngine.getTaskService().complete(task.getId());
                System.out.println("任务"+task.getId()+"已完成");
            }
        }
    }

    public void deleteAllTask(){
        List<Task> list = processEngine.getTaskService().createTaskQuery().list();
        //遍历任务列表
        if(list!=null&&list.size()>0){
            for(Task task:list){
                processEngine.getTaskService().deleteTask(task.getId());
                System.out.println("任务"+task.getId()+"已删除");
            }
        }
    }

    //查看bpmn 资源图片
    public void viewImage() throws Exception{
        String deploymentId="12501";
        String imageName=null;
        //取得某个部署的资源的名称  deploymentId
        List<String> resourceNames = processEngine.getRepositoryService().getDeploymentResourceNames(deploymentId);
        // buybill.bpmn  buybill.png
        if(resourceNames!=null&&resourceNames.size()>0){
            for(String temp :resourceNames){
                if(temp.indexOf(".png")>0){
                    imageName=temp;
                }
            }
        }

        /**
         * 读取资源
         * deploymentId:部署的id
         * resourceName：资源的文件名
         */
        InputStream resourceAsStream = processEngine.getRepositoryService()
                .getResourceAsStream(deploymentId, imageName);

        //把文件输入流写入到文件中
        File file=new File("d:/"+imageName);
        //FileUtils.copyInputStreamToFile(resourceAsStream, file);
    }



    //设置流程变量值
    public String setVariable(){
	    Scanner scanner = new Scanner(System.in);
        String taskId=scanner.nextLine();
        scanner.close();
        //采用TaskService来设置流程变量

        //1. 第一次设置流程变量
//        TaskService taskService = processEngine.getTaskService();
//        taskService.setVariable(taskId, "cost", 1000);//设置单一的变量，作用域在整个流程实例
//        taskService.setVariable(taskId, "申请时间", new Date());
//        taskService.setVariableLocal(taskId, "申请人", "何某某");//该变量只有在本任务中是有效的


        //2. 在不同的任务中设置变量
//        TaskService taskService = processEngine.getTaskService();
//        taskService.setVariable(taskId, "cost", 5000);//设置单一的变量，作用域在整个流程实例
//        taskService.setVariable(taskId, "申请时间", new Date());
//        taskService.setVariableLocal(taskId, "申请人", "李某某");//该变量只有在本任务中是有效的

        /**
         * 3. 变量支持的类型
         * - 简单的类型 ：String 、boolean、Integer、double、date
         * - 自定义对象bean
         */
        TaskService taskService = processEngine.getTaskService();
        //传递的一个自定义bean对象
        AppayBillBean appayBillBean=new AppayBillBean();
        appayBillBean.setId(1);
        appayBillBean.setCost(300);
        appayBillBean.setDate(new Date());
        appayBillBean.setAppayPerson("wjy");
        taskService.setVariable(taskId, "appayBillBean", appayBillBean);

        AppayBillBean appayBillBean1=new AppayBillBean();
        appayBillBean1.setId(4);
        appayBillBean1.setCost(9090);
        appayBillBean1.setDate(new Date());
        appayBillBean1.setAppayPerson("www");
        taskService.setVariableLocal(taskId, "appayBillBean1", appayBillBean1);

        System.out.println("设置成功！");
        return taskId;
    }


    //查询流程变量
    public void getVariable(){
	    String taskId = "70018";
        //读取实现序列化的对象变量数据
        TaskService taskService = processEngine.getTaskService();
        AppayBillBean appayBillBean=(AppayBillBean) taskService.getVariable(taskId, "appayBillBean");
        System.out.println("appayBillBean");
        System.out.println(appayBillBean.getCost());
        System.out.println(appayBillBean.getAppayPerson());

        AppayBillBean appayBillBean1=(AppayBillBean) taskService.getVariable(taskId, "appayBillBean1");
        System.out.println("appayBillBean1");
        System.out.println(appayBillBean1.getCost());
        System.out.println(appayBillBean1.getAppayPerson());

    }

    //添加用户
    public void addUser(){
	    IdentityService identityService = processEngine.getIdentityService();
        User user1 = identityService.newUser("cxy");
        user1.setFirstName("cheng");
        user1.setLastName("xiaoye");
        user1.setPassword("imreallysorry");
        user1.setEmail("123@qq.com");
        identityService.saveUser(user1);
        User user = identityService.createUserQuery().userId("cxy").singleResult();
        System.out.println("用户 ["+user.getFirstName()+"] 已添加");
    }

    public void queryAttachment(){
        //取得任务服务
        TaskService taskService = processEngine.getTaskService();
        //创建一个任务查询对象
        TaskQuery taskQuery = taskService.createTaskQuery();
        //办理人的任务列表
        List<Task> list = taskQuery.list();
        //遍历任务列表
        if(list!=null&&list.size()>0){
            for(Task task:list){
                if(task.getName().equals("Data Wrangling ")){
                    List<Attachment> attachmentList = taskService.getTaskAttachments(task.getId());
                    for(Attachment attachment:attachmentList){
                        System.out.println(task.getName()+"’s Attachment located in "+attachment.getUrl());
                    }
                }

            }
        }


    }


}
