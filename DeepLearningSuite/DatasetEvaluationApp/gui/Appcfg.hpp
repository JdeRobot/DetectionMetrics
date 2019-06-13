#ifndef APP_CONFIG
#define APP_CONFIG

#include "appconfig.h"
#include <QApplication>

class Appcfg {
public:
  Appcfg(int argc , char **argv);
  void exec();
  YAML::Node get_node();
  
private:
  QApplication *a;
  appconfig *w;
};

#endif
// int main(int argc, char *argv[])
// {
//   QApplication a(argc, argv);
//   MainWindow w;
//   w.show();
//   return a.exec();
// }
