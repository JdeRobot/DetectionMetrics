/********************************************************************************
** Form generated from reading UI file 'setclass.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SETCLASS_H
#define UI_SETCLASS_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SetClass
{
public:
    QWidget *centralWidget;
    QWidget *widget;
    QHBoxLayout *horizontalLayout;
    QLineEdit *lineEdit;
    QLabel *label;
    QComboBox *comboBox;
    QWidget *widget1;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *pushButton_ok;
    QPushButton *pushButton_cancel;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *SetClass)
    {
        if (SetClass->objectName().isEmpty())
            SetClass->setObjectName(QStringLiteral("SetClass"));
        SetClass->resize(400, 300);
        centralWidget = new QWidget(SetClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        widget = new QWidget(centralWidget);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(10, 70, 383, 29));
        horizontalLayout = new QHBoxLayout(widget);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        lineEdit = new QLineEdit(widget);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));

        horizontalLayout->addWidget(lineEdit);

        label = new QLabel(widget);
        label->setObjectName(QStringLiteral("label"));

        horizontalLayout->addWidget(label);

        comboBox = new QComboBox(widget);
        comboBox->setObjectName(QStringLiteral("comboBox"));

        horizontalLayout->addWidget(comboBox);

        widget1 = new QWidget(centralWidget);
        widget1->setObjectName(QStringLiteral("widget1"));
        widget1->setGeometry(QRect(170, 180, 178, 29));
        horizontalLayout_2 = new QHBoxLayout(widget1);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        pushButton_ok = new QPushButton(widget1);
        pushButton_ok->setObjectName(QStringLiteral("pushButton_ok"));

        horizontalLayout_2->addWidget(pushButton_ok);

        pushButton_cancel = new QPushButton(widget1);
        pushButton_cancel->setObjectName(QStringLiteral("pushButton_cancel"));

        horizontalLayout_2->addWidget(pushButton_cancel);

        SetClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(SetClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 400, 25));
        SetClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(SetClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        SetClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(SetClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        SetClass->setStatusBar(statusBar);

        retranslateUi(SetClass);

        QMetaObject::connectSlotsByName(SetClass);
    } // setupUi

    void retranslateUi(QMainWindow *SetClass)
    {
        SetClass->setWindowTitle(QApplication::translate("SetClass", "SetClass", Q_NULLPTR));
        label->setText(QApplication::translate("SetClass", "will be converted to ", Q_NULLPTR));
        pushButton_ok->setText(QApplication::translate("SetClass", "OK", Q_NULLPTR));
        pushButton_cancel->setText(QApplication::translate("SetClass", "CANCEL", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class SetClass: public Ui_SetClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SETCLASS_H
