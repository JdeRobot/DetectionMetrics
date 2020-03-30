/********************************************************************************
** Form generated from reading UI file 'pop_up.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_POP_UP_H
#define UI_POP_UP_H

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
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_pop_up
{
public:
    QWidget *centralWidget;
    QWidget *widget;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label;
    QHBoxLayout *horizontalLayout;
    QLineEdit *lineEdit;
    QToolButton *toolButton_1;
    QWidget *widget1;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *pushButton_ok;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *pop_up)
    {
        if (pop_up->objectName().isEmpty())
            pop_up->setObjectName(QStringLiteral("pop_up"));
        pop_up->resize(397, 253);
        centralWidget = new QWidget(pop_up);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        widget = new QWidget(centralWidget);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(50, 70, 253, 31));
        horizontalLayout_2 = new QHBoxLayout(widget);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        label = new QLabel(widget);
        label->setObjectName(QStringLiteral("label"));

        horizontalLayout_2->addWidget(label);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        lineEdit = new QLineEdit(widget);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));

        horizontalLayout->addWidget(lineEdit);

        toolButton_1 = new QToolButton(widget);
        toolButton_1->setObjectName(QStringLiteral("toolButton_1"));

        horizontalLayout->addWidget(toolButton_1);


        horizontalLayout_2->addLayout(horizontalLayout);

        widget1 = new QWidget(centralWidget);
        widget1->setObjectName(QStringLiteral("widget1"));
        widget1->setGeometry(QRect(230, 150, 121, 29));
        horizontalLayout_3 = new QHBoxLayout(widget1);
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        pushButton_ok = new QPushButton(widget1);
        pushButton_ok->setObjectName(QStringLiteral("pushButton_ok"));

        horizontalLayout_3->addWidget(pushButton_ok);

        pop_up->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(pop_up);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 397, 25));
        pop_up->setMenuBar(menuBar);
        mainToolBar = new QToolBar(pop_up);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        pop_up->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(pop_up);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        pop_up->setStatusBar(statusBar);

        retranslateUi(pop_up);

        QMetaObject::connectSlotsByName(pop_up);
    } // setupUi

    void retranslateUi(QMainWindow *pop_up)
    {
        pop_up->setWindowTitle(QApplication::translate("pop_up", "pop_up", Q_NULLPTR));
        label->setText(QApplication::translate("pop_up", "TextLabel", Q_NULLPTR));
        toolButton_1->setText(QApplication::translate("pop_up", "...", Q_NULLPTR));
        pushButton_ok->setText(QApplication::translate("pop_up", "OK", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class pop_up: public Ui_pop_up {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_POP_UP_H
