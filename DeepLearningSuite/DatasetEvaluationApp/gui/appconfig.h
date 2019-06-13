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
    void handleToolbuttonWeights();
    void handleToolbuttonNames();
    void handleToolbuttonCfg();
    void handleToolbuttonAppconfig();
    void handleToolbuttonEval();
    void handleCheckbox();
    void handlePushbuttonOK();

private:
    Ui::appconfig *ui;
    YAML::Node node;
};

#endif // APPCONFIG_H
