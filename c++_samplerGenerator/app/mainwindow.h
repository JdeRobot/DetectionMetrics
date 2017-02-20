#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <SampleGeneratorLib/Utils/SampleGenerationApp.h>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(SampleGenerationApp* app,QWidget *parent = 0);
    ~MainWindow();
    Ui::MainWindow *ui;


private:
    SampleGenerationApp* app;


private slots:
    void handleViewButton();
    void handleSelectOutputFolderButton();
    void handleSelectionNamesChanged();
    void setupTabsInformation();
    void handleConvertButton();
    void handleEvaluateButton();
    void handleDetectButton();
    void handleSelectOutputFolderButtonDetector();

};

#endif // MAINWINDOW_H
