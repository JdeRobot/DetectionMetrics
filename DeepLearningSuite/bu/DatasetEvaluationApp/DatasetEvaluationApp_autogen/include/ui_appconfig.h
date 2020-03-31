/********************************************************************************
** Form generated from reading UI file 'appconfig.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_APPCONFIG_H
#define UI_APPCONFIG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
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
    QHBoxLayout *horizontalLayout_12;
    QPushButton *pushButton_ok;
    QCheckBox *checkBox;
    QWidget *layoutWidget_2;
    QHBoxLayout *horizontalLayout_13;
    QVBoxLayout *verticalLayout_3;
    QLabel *label_6;
    QLabel *label_7;
    QLabel *label_8;
    QLabel *label_9;
    QVBoxLayout *verticalLayout_4;
    QHBoxLayout *horizontalLayout_3;
    QLineEdit *lineEdit_weights;
    QToolButton *toolButton_weights;
    QHBoxLayout *horizontalLayout_14;
    QLineEdit *lineEdit_eval;
    QToolButton *toolButton_eval;
    QHBoxLayout *horizontalLayout_15;
    QLineEdit *lineEdit_names;
    QToolButton *toolButton_names;
    QHBoxLayout *horizontalLayout_16;
    QLineEdit *lineEdit_cfg;
    QToolButton *toolButton_cfg;
    QWidget *layoutWidget_3;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_10;
    QLineEdit *lineEdit_appconfig;
    QToolButton *toolButton_appconfig;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *appconfig)
    {
        if (appconfig->objectName().isEmpty())
            appconfig->setObjectName(QStringLiteral("appconfig"));
        appconfig->resize(439, 333);
        centralWidget = new QWidget(appconfig);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        layoutWidget = new QWidget(centralWidget);
        layoutWidget->setObjectName(QStringLiteral("layoutWidget"));
        layoutWidget->setGeometry(QRect(230, 230, 178, 29));
        horizontalLayout_12 = new QHBoxLayout(layoutWidget);
        horizontalLayout_12->setSpacing(6);
        horizontalLayout_12->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_12->setObjectName(QStringLiteral("horizontalLayout_12"));
        horizontalLayout_12->setContentsMargins(0, 0, 0, 0);
        pushButton_ok = new QPushButton(layoutWidget);
        pushButton_ok->setObjectName(QStringLiteral("pushButton_ok"));

        horizontalLayout_12->addWidget(pushButton_ok);

        checkBox = new QCheckBox(centralWidget);
        checkBox->setObjectName(QStringLiteral("checkBox"));
        checkBox->setGeometry(QRect(130, 40, 161, 22));
        checkBox->setTristate(false);
        layoutWidget_2 = new QWidget(centralWidget);
        layoutWidget_2->setObjectName(QStringLiteral("layoutWidget_2"));
        layoutWidget_2->setGeometry(QRect(20, 70, 301, 141));
        horizontalLayout_13 = new QHBoxLayout(layoutWidget_2);
        horizontalLayout_13->setSpacing(6);
        horizontalLayout_13->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_13->setObjectName(QStringLiteral("horizontalLayout_13"));
        horizontalLayout_13->setContentsMargins(0, 0, 0, 0);
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        label_6 = new QLabel(layoutWidget_2);
        label_6->setObjectName(QStringLiteral("label_6"));

        verticalLayout_3->addWidget(label_6);

        label_7 = new QLabel(layoutWidget_2);
        label_7->setObjectName(QStringLiteral("label_7"));

        verticalLayout_3->addWidget(label_7);

        label_8 = new QLabel(layoutWidget_2);
        label_8->setObjectName(QStringLiteral("label_8"));

        verticalLayout_3->addWidget(label_8);

        label_9 = new QLabel(layoutWidget_2);
        label_9->setObjectName(QStringLiteral("label_9"));

        verticalLayout_3->addWidget(label_9);


        horizontalLayout_13->addLayout(verticalLayout_3);

        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        lineEdit_weights = new QLineEdit(layoutWidget_2);
        lineEdit_weights->setObjectName(QStringLiteral("lineEdit_weights"));
        lineEdit_weights->setEnabled(false);

        horizontalLayout_3->addWidget(lineEdit_weights);

        toolButton_weights = new QToolButton(layoutWidget_2);
        toolButton_weights->setObjectName(QStringLiteral("toolButton_weights"));
        toolButton_weights->setEnabled(false);

        horizontalLayout_3->addWidget(toolButton_weights);


        verticalLayout_4->addLayout(horizontalLayout_3);

        horizontalLayout_14 = new QHBoxLayout();
        horizontalLayout_14->setSpacing(6);
        horizontalLayout_14->setObjectName(QStringLiteral("horizontalLayout_14"));
        lineEdit_eval = new QLineEdit(layoutWidget_2);
        lineEdit_eval->setObjectName(QStringLiteral("lineEdit_eval"));
        lineEdit_eval->setEnabled(false);

        horizontalLayout_14->addWidget(lineEdit_eval);

        toolButton_eval = new QToolButton(layoutWidget_2);
        toolButton_eval->setObjectName(QStringLiteral("toolButton_eval"));
        toolButton_eval->setEnabled(false);

        horizontalLayout_14->addWidget(toolButton_eval);


        verticalLayout_4->addLayout(horizontalLayout_14);

        horizontalLayout_15 = new QHBoxLayout();
        horizontalLayout_15->setSpacing(6);
        horizontalLayout_15->setObjectName(QStringLiteral("horizontalLayout_15"));
        lineEdit_names = new QLineEdit(layoutWidget_2);
        lineEdit_names->setObjectName(QStringLiteral("lineEdit_names"));
        lineEdit_names->setEnabled(false);

        horizontalLayout_15->addWidget(lineEdit_names);

        toolButton_names = new QToolButton(layoutWidget_2);
        toolButton_names->setObjectName(QStringLiteral("toolButton_names"));
        toolButton_names->setEnabled(false);

        horizontalLayout_15->addWidget(toolButton_names);


        verticalLayout_4->addLayout(horizontalLayout_15);

        horizontalLayout_16 = new QHBoxLayout();
        horizontalLayout_16->setSpacing(6);
        horizontalLayout_16->setObjectName(QStringLiteral("horizontalLayout_16"));
        lineEdit_cfg = new QLineEdit(layoutWidget_2);
        lineEdit_cfg->setObjectName(QStringLiteral("lineEdit_cfg"));
        lineEdit_cfg->setEnabled(false);

        horizontalLayout_16->addWidget(lineEdit_cfg);

        toolButton_cfg = new QToolButton(layoutWidget_2);
        toolButton_cfg->setObjectName(QStringLiteral("toolButton_cfg"));
        toolButton_cfg->setEnabled(false);

        horizontalLayout_16->addWidget(toolButton_cfg);


        verticalLayout_4->addLayout(horizontalLayout_16);


        horizontalLayout_13->addLayout(verticalLayout_4);

        layoutWidget_3 = new QWidget(centralWidget);
        layoutWidget_3->setObjectName(QStringLiteral("layoutWidget_3"));
        layoutWidget_3->setGeometry(QRect(20, 10, 301, 29));
        horizontalLayout_4 = new QHBoxLayout(layoutWidget_3);
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(0, 0, 0, 0);
        label_10 = new QLabel(layoutWidget_3);
        label_10->setObjectName(QStringLiteral("label_10"));

        horizontalLayout_4->addWidget(label_10);

        lineEdit_appconfig = new QLineEdit(layoutWidget_3);
        lineEdit_appconfig->setObjectName(QStringLiteral("lineEdit_appconfig"));

        horizontalLayout_4->addWidget(lineEdit_appconfig);

        toolButton_appconfig = new QToolButton(layoutWidget_3);
        toolButton_appconfig->setObjectName(QStringLiteral("toolButton_appconfig"));

        horizontalLayout_4->addWidget(toolButton_appconfig);

        appconfig->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(appconfig);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 439, 25));
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
        appconfig->setWindowTitle(QApplication::translate("appconfig", "appconfig", Q_NULLPTR));
        pushButton_ok->setText(QApplication::translate("appconfig", "OK", Q_NULLPTR));
        checkBox->setText(QApplication::translate("appconfig", "Select Indivudually", Q_NULLPTR));
        label_6->setText(QApplication::translate("appconfig", "weightsPath", Q_NULLPTR));
        label_7->setText(QApplication::translate("appconfig", "evaluationsPath", Q_NULLPTR));
        label_8->setText(QApplication::translate("appconfig", "namesPath", Q_NULLPTR));
        label_9->setText(QApplication::translate("appconfig", "netCfgPath", Q_NULLPTR));
        toolButton_weights->setText(QApplication::translate("appconfig", "...", Q_NULLPTR));
        toolButton_eval->setText(QApplication::translate("appconfig", "...", Q_NULLPTR));
        toolButton_names->setText(QApplication::translate("appconfig", "...", Q_NULLPTR));
        toolButton_cfg->setText(QApplication::translate("appconfig", "...", Q_NULLPTR));
        label_10->setText(QApplication::translate("appconfig", "Config file", Q_NULLPTR));
        toolButton_appconfig->setText(QApplication::translate("appconfig", "...", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class appconfig: public Ui_appconfig {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_APPCONFIG_H
