package com.example.demo.Bean;

import java.io.Serializable;
import java.util.Date;

public class AppayBillBean implements Serializable {
    private Integer id;
    private Integer cost;//金额
    private String appayPerson;//申请人
    private Date date;//申请日期
    public Integer getId() {
        return id;
    }
    public void setId(Integer id) {
        this.id = id;
    }
    public Integer getCost() {
        return cost;
    }
    public void setCost(Integer cost) {
        this.cost = cost;
    }
    public String getAppayPerson() {
        return appayPerson;
    }
    public void setAppayPerson(String appayPerson) {
        this.appayPerson = appayPerson;
    }
    public Date getDate() {
        return date;
    }
    public void setDate(Date date) {
        this.date = date;
    }


}