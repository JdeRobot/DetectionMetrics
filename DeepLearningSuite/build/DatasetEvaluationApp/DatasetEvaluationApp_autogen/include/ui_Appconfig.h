/********************************************************************************
** Form generated from reading UI file 'Appconfig.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_APPCONFIG_H
#define UI_APPCONFIG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QWidget *widget;
    QHBoxLayout *horizontalLayout_10;
    QPushButton *pushButton_OK;
    QPushButton *pushButton_cancel;
    QWidget *widget1;
    QHBoxLayout *horizontalLayout_11;
    QVBoxLayout *verticalLayout_2;
    QLabel *label_2;
    QLabel *label;
    QLabel *label_4;
    QLabel *label_3;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QLineEdit *lineEdit1;
    QToolButton *toolButton1;
    QHBoxLayout *horizontalLayout_7;
    QLineEdit *lineEdit2;
    QToolButton *toolButton2;
    QHBoxLayout *horizontalLayout_8;
    QLineEdit *lineEdit3;
    QToolButton *toolButton3;
    QHBoxLayout *horizontalLayout_9;
    QLineEdit *lineEdit4;
    QToolButton *toolButton4;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(409, 299);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        widget = new QWidget(centralWidget);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(220, 210, 178, 29));
        horizontalLayout_10 = new QHBoxLayout(widget);
        horizontalLayout_10->setSpacing(6);
        horizontalLayout_10->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_10->setObjectName(QStringLiteral("horizontalLayout_10"));
        horizontalLayout_10->setContentsMargins(0, 0, 0, 0);
        pushButton_OK = new QPushButton(widget);
        pushButton_OK->setObjectName(QStringLiteral("pushButton_OK"));

        horizontalLayout_10->addWidget(pushButton_OK);

        pushButton_cancel = new QPushButton(widget);
        pushButton_cancel->setObjectName(QStringLiteral("pushButton_cancel"));

        horizontalLayout_10->addWidget(pushButton_cancel);

        widget1 = new QWidget(centralWidget);
        widget1->setObjectName(QStringLiteral("widget1"));
        widget1->setGeometry(QRect(21, 23, 301, 141));
        horizontalLayout_11 = new QHBoxLayout(widget1);
        horizontalLayout_11->setSpacing(6);
        horizontalLayout_11->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_11->setObjectName(QStringLiteral("horizontalLayout_11"));
        horizontalLayout_11->setContentsMargins(0, 0, 0, 0);
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        label_2 = new QLabel(widget1);
        label_2->setObjectName(QStringLiteral("label_2"));

        verticalLayout_2->addWidget(label_2);

        label = new QLabel(widget1);
        label->setObjectName(QStringLiteral("label"));

        verticalLayout_2->addWidget(label);

        label_4 = new QLabel(widget1);
        label_4->setObjectName(QStringLiteral("label_4"));

        verticalLayout_2->addWidget(label_4);

        label_3 = new QLabel(widget1);
        label_3->setObjectName(QStringLiteral("label_3"));

        verticalLayout_2->addWidget(label_3);


        horizontalLayout_11->addLayout(verticalLayout_2);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        lineEdit1 = new QLineEdit(widget1);
        lineEdit1->setObjectName(QStringLiteral("lineEdit1"));

        horizontalLayout->addWidget(lineEdit1);

        toolButton1 = new QToolButton(widget1);
        toolButton1->setObjectName(QStringLiteral("toolButton1"));

        horizontalLayout->addWidget(toolButton1);


        verticalLayout->addLayout(horizontalLayout);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(6);
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        lineEdit2 = new QLineEdit(widget1);
        lineEdit2->setObjectName(QStringLiteral("lineEdit2"));

        horizontalLayout_7->addWidget(lineEdit2);

        toolButton2 = new QToolButton(widget1);
        toolButton2->setObjectName(QStringLiteral("toolButton2"));

        horizontalLayout_7->addWidget(toolButton2);


        verticalLayout->addLayout(horizontalLayout_7);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setSpacing(6);
        horizontalLayout_8->setObjectName(QStringLiteral("horizontalLayout_8"));
        lineEdit3 = new QLineEdit(widget1);
        lineEdit3->setObjectName(QStringLiteral("lineEdit3"));

        horizontalLayout_8->addWidget(lineEdit3);

        toolButton3 = new QToolButton(widget1);
        toolButton3->setObjectName(QStringLiteral("toolButton3"));

        horizontalLayout_8->addWidget(toolButton3);


        verticalLayout->addLayout(horizontalLayout_8);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setSpacing(6);
        horizontalLayout_9->setObjectName(QStringLiteral("horizontalLayout_9"));
        lineEdit4 = new QLineEdit(widget1);
        lineEdit4->setObjectName(QStringLiteral("lineEdit4"));

        horizontalLayout_9->addWidget(lineEdit4);

        toolButton4 = new QToolButton(widget1);
        toolButton4->setObjectName(QStringLiteral("toolButton4"));

        horizontalLayout_9->addWidget(toolButton4);


        verticalLayout->addLayout(horizontalLayout_9);


        horizontalLayout_11->addLayout(verticalLayout);

        MainWindow->setCentralWidget(centralWidget);
        lineEdit1->raise();
        toolButton1->raise();
        label->raise();
        pushButton_OK->raise();
        pushButton_cancel->raise();
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 409, 25));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0));
        pushButton_OK->setText(QApplication::translate("MainWindow", "OK", 0));
        pushButton_cancel->setText(QApplication::translate("MainWindow", "CANCEL", 0));
        label_2->setText(QApplication::translate("MainWindow", "weightsPath", 0));
        label->setText(QApplication::translate("MainWindow", "evaluationsPath", 0));
        label_4->setText(QApplication::translate("MainWindow", "namesPath", 0));
        label_3->setText(QApplication::translate("MainWindow", "netCfgPath", 0));
        toolButton1->setText(QApplication::translate("MainWindow", "...", 0));
        toolButton2->setText(QApplication::translate("MainWindow", "...", 0));
        toolButton3->setText(QApplication::translate("MainWindow", "...", 0));
        toolButton4->setText(QApplication::translate("MainWindow", "...", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_APPCONFIG_H
