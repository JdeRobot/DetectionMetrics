/********************************************************************************
** Form generated from reading UI file 'appconfig.ui'
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

class Ui_appconfig
{
public:
    QWidget *centralWidget;
    QWidget *layoutWidget;
    QHBoxLayout *horizontalLayout_10;
    QPushButton *pushButton_OK;
    QPushButton *pushButton_cancel;
    QWidget *layoutWidget_2;
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

    void setupUi(QMainWindow *appconfig)
    {
        if (appconfig->objectName().isEmpty())
            appconfig->setObjectName(QStringLiteral("appconfig"));
        appconfig->resize(400, 300);
        centralWidget = new QWidget(appconfig);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        layoutWidget = new QWidget(centralWidget);
        layoutWidget->setObjectName(QStringLiteral("layoutWidget"));
        layoutWidget->setGeometry(QRect(209, 197, 178, 29));
        horizontalLayout_10 = new QHBoxLayout(layoutWidget);
        horizontalLayout_10->setSpacing(6);
        horizontalLayout_10->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_10->setObjectName(QStringLiteral("horizontalLayout_10"));
        horizontalLayout_10->setContentsMargins(0, 0, 0, 0);
        pushButton_OK = new QPushButton(layoutWidget);
        pushButton_OK->setObjectName(QStringLiteral("pushButton_OK"));

        horizontalLayout_10->addWidget(pushButton_OK);

        pushButton_cancel = new QPushButton(layoutWidget);
        pushButton_cancel->setObjectName(QStringLiteral("pushButton_cancel"));

        horizontalLayout_10->addWidget(pushButton_cancel);

        layoutWidget_2 = new QWidget(centralWidget);
        layoutWidget_2->setObjectName(QStringLiteral("layoutWidget_2"));
        layoutWidget_2->setGeometry(QRect(30, 20, 301, 141));
        horizontalLayout_11 = new QHBoxLayout(layoutWidget_2);
        horizontalLayout_11->setSpacing(6);
        horizontalLayout_11->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_11->setObjectName(QStringLiteral("horizontalLayout_11"));
        horizontalLayout_11->setContentsMargins(0, 0, 0, 0);
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        label_2 = new QLabel(layoutWidget_2);
        label_2->setObjectName(QStringLiteral("label_2"));

        verticalLayout_2->addWidget(label_2);

        label = new QLabel(layoutWidget_2);
        label->setObjectName(QStringLiteral("label"));

        verticalLayout_2->addWidget(label);

        label_4 = new QLabel(layoutWidget_2);
        label_4->setObjectName(QStringLiteral("label_4"));

        verticalLayout_2->addWidget(label_4);

        label_3 = new QLabel(layoutWidget_2);
        label_3->setObjectName(QStringLiteral("label_3"));

        verticalLayout_2->addWidget(label_3);


        horizontalLayout_11->addLayout(verticalLayout_2);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        lineEdit1 = new QLineEdit(layoutWidget_2);
        lineEdit1->setObjectName(QStringLiteral("lineEdit1"));

        horizontalLayout->addWidget(lineEdit1);

        toolButton1 = new QToolButton(layoutWidget_2);
        toolButton1->setObjectName(QStringLiteral("toolButton1"));

        horizontalLayout->addWidget(toolButton1);


        verticalLayout->addLayout(horizontalLayout);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(6);
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        lineEdit2 = new QLineEdit(layoutWidget_2);
        lineEdit2->setObjectName(QStringLiteral("lineEdit2"));

        horizontalLayout_7->addWidget(lineEdit2);

        toolButton2 = new QToolButton(layoutWidget_2);
        toolButton2->setObjectName(QStringLiteral("toolButton2"));

        horizontalLayout_7->addWidget(toolButton2);


        verticalLayout->addLayout(horizontalLayout_7);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setSpacing(6);
        horizontalLayout_8->setObjectName(QStringLiteral("horizontalLayout_8"));
        lineEdit3 = new QLineEdit(layoutWidget_2);
        lineEdit3->setObjectName(QStringLiteral("lineEdit3"));

        horizontalLayout_8->addWidget(lineEdit3);

        toolButton3 = new QToolButton(layoutWidget_2);
        toolButton3->setObjectName(QStringLiteral("toolButton3"));

        horizontalLayout_8->addWidget(toolButton3);


        verticalLayout->addLayout(horizontalLayout_8);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setSpacing(6);
        horizontalLayout_9->setObjectName(QStringLiteral("horizontalLayout_9"));
        lineEdit4 = new QLineEdit(layoutWidget_2);
        lineEdit4->setObjectName(QStringLiteral("lineEdit4"));

        horizontalLayout_9->addWidget(lineEdit4);

        toolButton4 = new QToolButton(layoutWidget_2);
        toolButton4->setObjectName(QStringLiteral("toolButton4"));

        horizontalLayout_9->addWidget(toolButton4);


        verticalLayout->addLayout(horizontalLayout_9);


        horizontalLayout_11->addLayout(verticalLayout);

        appconfig->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(appconfig);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 400, 25));
        appconfig->setMenuBar(menuBar);
        mainToolBar = new QToolBar(appconfig);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        appconfig->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(appconfig);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        appconfig->setStatusBar(statusBar);

        retranslateUi(appconfig);

        QMetaObject::connectSlotsByName(appconfig);
    } // setupUi

    void retranslateUi(QMainWindow *appconfig)
    {
        appconfig->setWindowTitle(QApplication::translate("appconfig", "appconfig", 0));
        pushButton_OK->setText(QApplication::translate("appconfig", "OK", 0));
        pushButton_cancel->setText(QApplication::translate("appconfig", "CANCEL", 0));
        label_2->setText(QApplication::translate("appconfig", "weightsPath", 0));
        label->setText(QApplication::translate("appconfig", "evaluationsPath", 0));
        label_4->setText(QApplication::translate("appconfig", "namesPath", 0));
        label_3->setText(QApplication::translate("appconfig", "netCfgPath", 0));
        toolButton1->setText(QApplication::translate("appconfig", "...", 0));
        toolButton2->setText(QApplication::translate("appconfig", "...", 0));
        toolButton3->setText(QApplication::translate("appconfig", "...", 0));
        toolButton4->setText(QApplication::translate("appconfig", "...", 0));
    } // retranslateUi

};

namespace Ui {
    class appconfig: public Ui_appconfig {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_APPCONFIG_H
