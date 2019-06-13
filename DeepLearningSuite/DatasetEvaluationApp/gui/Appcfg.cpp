#include "Appcfg.hpp"
#include <iostream>
Appcfg::Appcfg(int argc, char **argv){
    this->a = new QApplication(argc,argv);
    this->w = new appconfig();
    Appcfg::exec();
}

void Appcfg::exec(){
  this->w->show();
  this->a->exec();
}

YAML::Node Appcfg::get_node(){
  return this->w->return_node();
}
