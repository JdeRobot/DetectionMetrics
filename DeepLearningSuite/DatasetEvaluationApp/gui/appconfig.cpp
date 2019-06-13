#include "appconfig.h"
#include "ui_appconfig.h"
#include <QFileDialog>
#include <iostream>

appconfig::appconfig(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::appconfig)
{
    ui->setupUi(this);
    this->node["datasetPath"]="~/";
}

appconfig::~appconfig(){
    delete ui;
}

void appconfig::on_toolButton1_clicked(){
    QString dir_name = QFileDialog::getExistingDirectory(this,"Open a dir","~/");
//    std::string utf8_text = dir_name.toUtf8().constData();
    ui->lineEdit1->setText(dir_name);
    this->node["weightsPath"]=dir_name.toUtf8().constData();
}

void appconfig::on_toolButton2_clicked(){
    QString dir_name = QFileDialog::getExistingDirectory(this,"Open a dir","~/");
//    std::string utf8_text = dir_name.toUtf8().constData();
    ui->lineEdit2->setText(dir_name);
    this->node["evaluationsPath"]=dir_name.toUtf8().constData();
}

void appconfig::on_toolButton3_clicked(){
    QString dir_name = QFileDialog::getExistingDirectory(this,"Open a dir","~/");
//    std::string utf8_text = dir_name.toUtf8().constData();
    ui->lineEdit3->setText(dir_name);
    this->node["namesPath"]=dir_name.toUtf8().constData();
}

void appconfig::on_toolButton4_clicked(){
    QString dir_name = QFileDialog::getExistingDirectory(this,"Open a dir","~/");
//    std::string utf8_text = dir_name.toUtf8().constData();
    ui->lineEdit4->setText(dir_name);
    this->node["netCfgPath"]=dir_name.toUtf8().constData();
}

YAML::Node appconfig::return_node(){
  return this->node;
}
