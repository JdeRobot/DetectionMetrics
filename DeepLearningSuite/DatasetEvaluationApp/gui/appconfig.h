#ifndef APPCONFIG_H
#define APPCONFIG_H

#include <QMainWindow>
#include <yaml-cpp/yaml.h>

namespace Ui {
class appconfig;
}

class appconfig : public QMainWindow
{
    Q_OBJECT

public:
    explicit appconfig(QWidget *parent = 0);
    ~appconfig();
    YAML::Node return_node();
private slots:
    void on_toolButton1_clicked();
    void on_toolButton2_clicked();
    void on_toolButton3_clicked();
    void on_toolButton4_clicked();
private:
    Ui::appconfig *ui;
    YAML::Node node;
};

#endif // APPCONFIG_H
