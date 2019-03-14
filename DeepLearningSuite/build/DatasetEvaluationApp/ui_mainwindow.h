/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QListView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QTabWidget *tabWidget;
    QWidget *tab_1;
    QPushButton *pushButton;
    QWidget *verticalLayoutWidget;
    QVBoxLayout *verticalLayout;
    QLabel *label_1111_;
    QListView *listView_viewer_dataset;
    QWidget *verticalLayoutWidget_2;
    QVBoxLayout *verticalLayout_2;
    QLabel *label_27;
    QListView *listView_viewer_names;
    QWidget *verticalLayoutWidget_3;
    QVBoxLayout *verticalLayout_3;
    QLabel *label_31;
    QListView *listView_viewer_reader_imp;
    QWidget *verticalLayoutWidget_4;
    QVBoxLayout *verticalLayout_4;
    QLabel *label_41;
    QListView *listView_viewer_classFilter;
    QCheckBox *checkBox_evaluator_show_depth;
    QWidget *tab_2;
    QWidget *verticalLayoutWidget_5;
    QVBoxLayout *verticalLayout_5;
    QLabel *label_5;
    QListView *listView_converter_reader_imp;
    QWidget *verticalLayoutWidget_6;
    QVBoxLayout *verticalLayout_6;
    QLabel *label_6;
    QListView *listView_converter_names;
    QWidget *verticalLayoutWidget_7;
    QVBoxLayout *verticalLayout_7;
    QLabel *label_7;
    QListView *listView_converter_dataset;
    QWidget *verticalLayoutWidget_8;
    QVBoxLayout *verticalLayout_8;
    QLabel *label_8;
    QListView *listView_converter_classFilter;
    QWidget *verticalLayoutWidget_9;
    QVBoxLayout *verticalLayout_9;
    QLabel *label_9;
    QListView *listView_converter_outImp;
    QWidget *verticalLayoutWidget_101;
    QVBoxLayout *verticalLayout_101;
    QLabel *label_101;
    QListView *listView_converter_writer_names;
    QCheckBox *checkBox_use_writernames;
    QTextEdit *textEdit_converterOutPath;
    QLabel *label_11;
    QPushButton *pushButton_converter_output;
    QPushButton *pushButton_convert;
    QCheckBox *checkBox_splitActive;
    QLabel *label_10;
    QTextEdit *textEdit_converter_trainRatio;
    QWidget *verticalLayoutWidget_10;
    QVBoxLayout *verticalLayout_10;
    QLabel *label_12;
    QFrame *line;
    QCheckBox *checkBox_converter_write_images;
    QWidget *tab;
    QWidget *verticalLayoutWidget_18;
    QVBoxLayout *verticalLayout_18;
    QLabel *label_20;
    QListView *listView_detector_reader_imp;
    QWidget *verticalLayoutWidget_19;
    QVBoxLayout *verticalLayout_19;
    QLabel *label_21;
    QListView *listView_detector_names;
    QWidget *verticalLayoutWidget_20;
    QVBoxLayout *verticalLayout_20;
    QLabel *label_22;
    QListView *listView_detector_dataset;
    QTextEdit *textEdit_detectorOutPath;
    QPushButton *pushButton_detector_output;
    QWidget *verticalLayoutWidget_21;
    QVBoxLayout *verticalLayout_21;
    QLabel *label_23;
    QListView *listView_detector_weights;
    QWidget *verticalLayoutWidget_22;
    QVBoxLayout *verticalLayout_22;
    QLabel *label_24;
    QListView *listView_detector_net_config;
    QWidget *verticalLayoutWidget_23;
    QVBoxLayout *verticalLayout_23;
    QLabel *label_25;
    QListView *listView_detector_imp;
    QPushButton *pushButton_detect;
    QCheckBox *checkBox_detector_useDepth;
    QCheckBox *checkBox_detector_single;
    QWidget *verticalLayoutWidget_24;
    QVBoxLayout *verticalLayout_24;
    QLabel *label_26;
    QListView *listView_detector_names_inferencer;
    QGroupBox *detector_groupBox_inferencer_params;
    QCheckBox *detector_checkBox_use_rgb;
    QLineEdit *detector_lineEdit_confidence_thresh;
    QLabel *label;
    QLabel *label_211;
    QLineEdit *detector_lineEdit_inferencer_scaling_factor;
    QLineEdit *detector_lineEdit_mean_sub_blue;
    QLineEdit *detector_lineEdit_mean_sub_green;
    QLineEdit *detector_lineEdit_mean_sub_red;
    QLabel *label_33;
    QLabel *label_43;
    QLineEdit *detector_lineEdit_inferencer_input_width;
    QLineEdit *detector_lineEdit_inferencer_input_height;
    QWidget *tab_3;
    QWidget *verticalLayoutWidget_11;
    QVBoxLayout *verticalLayout_11;
    QLabel *label_13;
    QListView *listView_evaluator_gt_names;
    QWidget *verticalLayoutWidget_12;
    QVBoxLayout *verticalLayout_12;
    QLabel *label_14;
    QListView *listView_evaluator_gt_dataset;
    QWidget *verticalLayoutWidget_13;
    QVBoxLayout *verticalLayout_13;
    QLabel *label_15;
    QListView *listView_evaluator_gt_imp;
    QWidget *verticalLayoutWidget_14;
    QVBoxLayout *verticalLayout_14;
    QLabel *label_16;
    QListView *listView_evaluator_detection_names;
    QWidget *verticalLayoutWidget_15;
    QVBoxLayout *verticalLayout_15;
    QLabel *label_17;
    QListView *listView_evaluator_dectection_dataset;
    QWidget *verticalLayoutWidget_16;
    QVBoxLayout *verticalLayout_16;
    QLabel *label_18;
    QListView *listView_evaluator_detection_imp;
    QWidget *verticalLayoutWidget_17;
    QVBoxLayout *verticalLayout_17;
    QLabel *label_19;
    QListView *listView_evaluator_classFilter;
    QPushButton *pushButton_evaluate;
    QCheckBox *checkBox_evaluator_merge;
    QCheckBox *checkBox_evaluator_mix;
    QGroupBox *evaluator_ioutype_groupbox;
    QRadioButton *radioButton_evaluator_iou_bbox;
    QRadioButton *radioButton_evaluator_iou_seg;
    QWidget *tab_4;
    QWidget *verticalLayoutWidget_25;
    QVBoxLayout *verticalLayout_25;
    QLabel *label_271;
    QListView *listView_deploy_impl;
    QWidget *verticalLayoutWidget_26;
    QVBoxLayout *verticalLayout_26;
    QLabel *label_28;
    QListView *listView_deploy_names_inferencer;
    QWidget *verticalLayoutWidget_28;
    QVBoxLayout *verticalLayout_28;
    QLabel *label_30;
    QListView *listView_deploy_weights;
    QLabel *label_311;
    QListView *listView_deploy_input_imp;
    QGroupBox *deployer_param_groupBox;
    QLabel *label21;
    QLineEdit *lineEdit_deployer_proxy;
    QLineEdit *lineEdit_deployer_format;
    QLineEdit *lineEdit_deployer_topic;
    QLineEdit *lineEdit_deployer_name;
    QLabel *label_2;
    QLabel *label_3;
    QLabel *label_4;
    QRadioButton *radioButton_deployer_ros;
    QRadioButton *radioButton_deployer_ice;
    QLabel *label_51;
    QGroupBox *groupBox_config_option;
    QRadioButton *deployer_radioButton_manual;
    QRadioButton *deployer_radioButton_config;
    QTextEdit *textEdit_deployInputPath;
    QPushButton *pushButton_deploy_input;
    QPushButton *pushButton_deploy_process;
    QGroupBox *deployer_groupBox_inferencer_params;
    QCheckBox *deployer_checkBox_use_rgb;
    QLabel *label_210;
    QLineEdit *deployer_lineEdit_inferencer_scaling_factor;
    QLineEdit *deployer_lineEdit_mean_sub_blue;
    QLineEdit *deployer_lineEdit_mean_sub_green;
    QLineEdit *deployer_lineEdit_mean_sub_red;
    QLabel *label_32;
    QLabel *label_42;
    QLineEdit *deployer_lineEdit_inferencer_input_width;
    QLineEdit *deployer_lineEdit_inferencer_input_height;
    QGroupBox *groupbox_deployer_saveOutput;
    QPushButton *pushButton_deployer_output_folder;
    QTextEdit *textEdit_deployerOutputPath;
    QCheckBox *checkBox_deployer_saveOutput;
    QWidget *verticalLayoutWidget_27;
    QVBoxLayout *verticalLayout_27;
    QLabel *label_29;
    QListView *listView_deploy_net_config;
    QPushButton *pushButton_stop_deployer_process;
    QSlider *deployer_conf_horizontalSlider;
    QLabel *deployer_confidence_label;
    QLineEdit *deployer_confidence_lineEdit;
    QGroupBox *deployer_cameraID_groupBox;
    QSpinBox *deployer_camera_spinBox;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1225, 764);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(0, 0, 1221, 701));
        tab_1 = new QWidget();
        tab_1->setObjectName(QStringLiteral("tab_1"));
        pushButton = new QPushButton(tab_1);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(1040, 20, 85, 28));
        verticalLayoutWidget = new QWidget(tab_1);
        verticalLayoutWidget->setObjectName(QStringLiteral("verticalLayoutWidget"));
        verticalLayoutWidget->setGeometry(QRect(10, 10, 411, 291));
        verticalLayout = new QVBoxLayout(verticalLayoutWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        label_1111_ = new QLabel(verticalLayoutWidget);
        label_1111_->setObjectName(QStringLiteral("label_1111_"));

        verticalLayout->addWidget(label_1111_);

        listView_viewer_dataset = new QListView(verticalLayoutWidget);
        listView_viewer_dataset->setObjectName(QStringLiteral("listView_viewer_dataset"));

        verticalLayout->addWidget(listView_viewer_dataset);

        verticalLayoutWidget_2 = new QWidget(tab_1);
        verticalLayoutWidget_2->setObjectName(QStringLiteral("verticalLayoutWidget_2"));
        verticalLayoutWidget_2->setGeometry(QRect(440, 10, 191, 131));
        verticalLayout_2 = new QVBoxLayout(verticalLayoutWidget_2);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        label_27 = new QLabel(verticalLayoutWidget_2);
        label_27->setObjectName(QStringLiteral("label_27"));

        verticalLayout_2->addWidget(label_27);

        listView_viewer_names = new QListView(verticalLayoutWidget_2);
        listView_viewer_names->setObjectName(QStringLiteral("listView_viewer_names"));

        verticalLayout_2->addWidget(listView_viewer_names);

        verticalLayoutWidget_3 = new QWidget(tab_1);
        verticalLayoutWidget_3->setObjectName(QStringLiteral("verticalLayoutWidget_3"));
        verticalLayoutWidget_3->setGeometry(QRect(650, 10, 168, 131));
        verticalLayout_3 = new QVBoxLayout(verticalLayoutWidget_3);
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        label_31 = new QLabel(verticalLayoutWidget_3);
        label_31->setObjectName(QStringLiteral("label_31"));

        verticalLayout_3->addWidget(label_31);

        listView_viewer_reader_imp = new QListView(verticalLayoutWidget_3);
        listView_viewer_reader_imp->setObjectName(QStringLiteral("listView_viewer_reader_imp"));

        verticalLayout_3->addWidget(listView_viewer_reader_imp);

        verticalLayoutWidget_4 = new QWidget(tab_1);
        verticalLayoutWidget_4->setObjectName(QStringLiteral("verticalLayoutWidget_4"));
        verticalLayoutWidget_4->setGeometry(QRect(10, 320, 411, 321));
        verticalLayout_4 = new QVBoxLayout(verticalLayoutWidget_4);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(0, 0, 0, 0);
        label_41 = new QLabel(verticalLayoutWidget_4);
        label_41->setObjectName(QStringLiteral("label_41"));

        verticalLayout_4->addWidget(label_41);

        listView_viewer_classFilter = new QListView(verticalLayoutWidget_4);
        listView_viewer_classFilter->setObjectName(QStringLiteral("listView_viewer_classFilter"));

        verticalLayout_4->addWidget(listView_viewer_classFilter);

        checkBox_evaluator_show_depth = new QCheckBox(tab_1);
        checkBox_evaluator_show_depth->setObjectName(QStringLiteral("checkBox_evaluator_show_depth"));
        checkBox_evaluator_show_depth->setGeometry(QRect(770, 150, 191, 26));
        checkBox_evaluator_show_depth->setChecked(false);
        tabWidget->addTab(tab_1, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        verticalLayoutWidget_5 = new QWidget(tab_2);
        verticalLayoutWidget_5->setObjectName(QStringLiteral("verticalLayoutWidget_5"));
        verticalLayoutWidget_5->setGeometry(QRect(650, 10, 220, 131));
        verticalLayout_5 = new QVBoxLayout(verticalLayoutWidget_5);
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setContentsMargins(11, 11, 11, 11);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(0, 0, 0, 0);
        label_5 = new QLabel(verticalLayoutWidget_5);
        label_5->setObjectName(QStringLiteral("label_5"));

        verticalLayout_5->addWidget(label_5);

        listView_converter_reader_imp = new QListView(verticalLayoutWidget_5);
        listView_converter_reader_imp->setObjectName(QStringLiteral("listView_converter_reader_imp"));

        verticalLayout_5->addWidget(listView_converter_reader_imp);

        verticalLayoutWidget_6 = new QWidget(tab_2);
        verticalLayoutWidget_6->setObjectName(QStringLiteral("verticalLayoutWidget_6"));
        verticalLayoutWidget_6->setGeometry(QRect(440, 10, 191, 131));
        verticalLayout_6 = new QVBoxLayout(verticalLayoutWidget_6);
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setContentsMargins(11, 11, 11, 11);
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        verticalLayout_6->setContentsMargins(0, 0, 0, 0);
        label_6 = new QLabel(verticalLayoutWidget_6);
        label_6->setObjectName(QStringLiteral("label_6"));

        verticalLayout_6->addWidget(label_6);

        listView_converter_names = new QListView(verticalLayoutWidget_6);
        listView_converter_names->setObjectName(QStringLiteral("listView_converter_names"));

        verticalLayout_6->addWidget(listView_converter_names);

        verticalLayoutWidget_7 = new QWidget(tab_2);
        verticalLayoutWidget_7->setObjectName(QStringLiteral("verticalLayoutWidget_7"));
        verticalLayoutWidget_7->setGeometry(QRect(10, 10, 411, 291));
        verticalLayout_7 = new QVBoxLayout(verticalLayoutWidget_7);
        verticalLayout_7->setSpacing(6);
        verticalLayout_7->setContentsMargins(11, 11, 11, 11);
        verticalLayout_7->setObjectName(QStringLiteral("verticalLayout_7"));
        verticalLayout_7->setContentsMargins(0, 0, 0, 0);
        label_7 = new QLabel(verticalLayoutWidget_7);
        label_7->setObjectName(QStringLiteral("label_7"));

        verticalLayout_7->addWidget(label_7);

        listView_converter_dataset = new QListView(verticalLayoutWidget_7);
        listView_converter_dataset->setObjectName(QStringLiteral("listView_converter_dataset"));

        verticalLayout_7->addWidget(listView_converter_dataset);

        verticalLayoutWidget_8 = new QWidget(tab_2);
        verticalLayoutWidget_8->setObjectName(QStringLiteral("verticalLayoutWidget_8"));
        verticalLayoutWidget_8->setGeometry(QRect(10, 320, 411, 321));
        verticalLayout_8 = new QVBoxLayout(verticalLayoutWidget_8);
        verticalLayout_8->setSpacing(6);
        verticalLayout_8->setContentsMargins(11, 11, 11, 11);
        verticalLayout_8->setObjectName(QStringLiteral("verticalLayout_8"));
        verticalLayout_8->setContentsMargins(0, 0, 0, 0);
        label_8 = new QLabel(verticalLayoutWidget_8);
        label_8->setObjectName(QStringLiteral("label_8"));

        verticalLayout_8->addWidget(label_8);

        listView_converter_classFilter = new QListView(verticalLayoutWidget_8);
        listView_converter_classFilter->setObjectName(QStringLiteral("listView_converter_classFilter"));

        verticalLayout_8->addWidget(listView_converter_classFilter);

        verticalLayoutWidget_9 = new QWidget(tab_2);
        verticalLayoutWidget_9->setObjectName(QStringLiteral("verticalLayoutWidget_9"));
        verticalLayoutWidget_9->setGeometry(QRect(510, 220, 216, 111));
        verticalLayout_9 = new QVBoxLayout(verticalLayoutWidget_9);
        verticalLayout_9->setSpacing(6);
        verticalLayout_9->setContentsMargins(11, 11, 11, 11);
        verticalLayout_9->setObjectName(QStringLiteral("verticalLayout_9"));
        verticalLayout_9->setContentsMargins(0, 0, 0, 0);
        label_9 = new QLabel(verticalLayoutWidget_9);
        label_9->setObjectName(QStringLiteral("label_9"));

        verticalLayout_9->addWidget(label_9);

        listView_converter_outImp = new QListView(verticalLayoutWidget_9);
        listView_converter_outImp->setObjectName(QStringLiteral("listView_converter_outImp"));

        verticalLayout_9->addWidget(listView_converter_outImp);

        verticalLayoutWidget_101 = new QWidget(tab_2);
        verticalLayoutWidget_101->setObjectName(QStringLiteral("verticalLayoutWidget_101"));
        verticalLayoutWidget_101->setGeometry(QRect(750, 210, 191, 131));
        verticalLayout_101 = new QVBoxLayout(verticalLayoutWidget_101);
        verticalLayout_101->setSpacing(6);
        verticalLayout_101->setContentsMargins(11, 11, 11, 11);
        verticalLayout_101->setObjectName(QStringLiteral("verticalLayout_101"));
        verticalLayout_101->setContentsMargins(0, 0, 0, 0);
        label_101 = new QLabel(verticalLayoutWidget_101);
        label_101->setObjectName(QStringLiteral("label_101"));

        verticalLayout_101->addWidget(label_101);

        listView_converter_writer_names = new QListView(verticalLayoutWidget_101);
        listView_converter_writer_names->setObjectName(QStringLiteral("listView_converter_writer_names"));

        verticalLayout_101->addWidget(listView_converter_writer_names);

        checkBox_use_writernames = new QCheckBox(tab_2);
        checkBox_use_writernames->setObjectName(QStringLiteral("checkBox_use_writernames"));
        checkBox_use_writernames->setGeometry(QRect(870, 350, 341, 26));
        checkBox_use_writernames->setChecked(false);
        textEdit_converterOutPath = new QTextEdit(tab_2);
        textEdit_converterOutPath->setObjectName(QStringLiteral("textEdit_converterOutPath"));
        textEdit_converterOutPath->setGeometry(QRect(510, 500, 401, 21));
        label_11 = new QLabel(tab_2);
        label_11->setObjectName(QStringLiteral("label_11"));
        label_11->setGeometry(QRect(510, 440, 204, 20));
        pushButton_converter_output = new QPushButton(tab_2);
        pushButton_converter_output->setObjectName(QStringLiteral("pushButton_converter_output"));
        pushButton_converter_output->setGeometry(QRect(510, 460, 141, 28));
        pushButton_convert = new QPushButton(tab_2);
        pushButton_convert->setObjectName(QStringLiteral("pushButton_convert"));
        pushButton_convert->setGeometry(QRect(1040, 50, 85, 28));
        checkBox_splitActive = new QCheckBox(tab_2);
        checkBox_splitActive->setObjectName(QStringLiteral("checkBox_splitActive"));
        checkBox_splitActive->setGeometry(QRect(750, 400, 181, 26));
        label_10 = new QLabel(tab_2);
        label_10->setObjectName(QStringLiteral("label_10"));
        label_10->setGeometry(QRect(750, 430, 81, 20));
        textEdit_converter_trainRatio = new QTextEdit(tab_2);
        textEdit_converter_trainRatio->setObjectName(QStringLiteral("textEdit_converter_trainRatio"));
        textEdit_converter_trainRatio->setGeometry(QRect(750, 450, 171, 21));
        verticalLayoutWidget_10 = new QWidget(tab_2);
        verticalLayoutWidget_10->setObjectName(QStringLiteral("verticalLayoutWidget_10"));
        verticalLayoutWidget_10->setGeometry(QRect(510, 540, 178, 95));
        verticalLayout_10 = new QVBoxLayout(verticalLayoutWidget_10);
        verticalLayout_10->setSpacing(6);
        verticalLayout_10->setContentsMargins(11, 11, 11, 11);
        verticalLayout_10->setObjectName(QStringLiteral("verticalLayout_10"));
        verticalLayout_10->setContentsMargins(0, 0, 0, 0);
        label_12 = new QLabel(verticalLayoutWidget_10);
        label_12->setObjectName(QStringLiteral("label_12"));

        verticalLayout_10->addWidget(label_12);

        line = new QFrame(verticalLayoutWidget_10);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        verticalLayout_10->addWidget(line);

        checkBox_converter_write_images = new QCheckBox(verticalLayoutWidget_10);
        checkBox_converter_write_images->setObjectName(QStringLiteral("checkBox_converter_write_images"));

        verticalLayout_10->addWidget(checkBox_converter_write_images);

        tabWidget->addTab(tab_2, QString());
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        verticalLayoutWidget_18 = new QWidget(tab);
        verticalLayoutWidget_18->setObjectName(QStringLiteral("verticalLayoutWidget_18"));
        verticalLayoutWidget_18->setGeometry(QRect(650, 10, 168, 131));
        verticalLayout_18 = new QVBoxLayout(verticalLayoutWidget_18);
        verticalLayout_18->setSpacing(6);
        verticalLayout_18->setContentsMargins(11, 11, 11, 11);
        verticalLayout_18->setObjectName(QStringLiteral("verticalLayout_18"));
        verticalLayout_18->setContentsMargins(0, 0, 0, 0);
        label_20 = new QLabel(verticalLayoutWidget_18);
        label_20->setObjectName(QStringLiteral("label_20"));

        verticalLayout_18->addWidget(label_20);

        listView_detector_reader_imp = new QListView(verticalLayoutWidget_18);
        listView_detector_reader_imp->setObjectName(QStringLiteral("listView_detector_reader_imp"));

        verticalLayout_18->addWidget(listView_detector_reader_imp);

        verticalLayoutWidget_19 = new QWidget(tab);
        verticalLayoutWidget_19->setObjectName(QStringLiteral("verticalLayoutWidget_19"));
        verticalLayoutWidget_19->setGeometry(QRect(440, 10, 191, 131));
        verticalLayout_19 = new QVBoxLayout(verticalLayoutWidget_19);
        verticalLayout_19->setSpacing(6);
        verticalLayout_19->setContentsMargins(11, 11, 11, 11);
        verticalLayout_19->setObjectName(QStringLiteral("verticalLayout_19"));
        verticalLayout_19->setContentsMargins(0, 0, 0, 0);
        label_21 = new QLabel(verticalLayoutWidget_19);
        label_21->setObjectName(QStringLiteral("label_21"));

        verticalLayout_19->addWidget(label_21);

        listView_detector_names = new QListView(verticalLayoutWidget_19);
        listView_detector_names->setObjectName(QStringLiteral("listView_detector_names"));

        verticalLayout_19->addWidget(listView_detector_names);

        verticalLayoutWidget_20 = new QWidget(tab);
        verticalLayoutWidget_20->setObjectName(QStringLiteral("verticalLayoutWidget_20"));
        verticalLayoutWidget_20->setGeometry(QRect(10, 10, 411, 291));
        verticalLayout_20 = new QVBoxLayout(verticalLayoutWidget_20);
        verticalLayout_20->setSpacing(6);
        verticalLayout_20->setContentsMargins(11, 11, 11, 11);
        verticalLayout_20->setObjectName(QStringLiteral("verticalLayout_20"));
        verticalLayout_20->setContentsMargins(0, 0, 0, 0);
        label_22 = new QLabel(verticalLayoutWidget_20);
        label_22->setObjectName(QStringLiteral("label_22"));

        verticalLayout_20->addWidget(label_22);

        listView_detector_dataset = new QListView(verticalLayoutWidget_20);
        listView_detector_dataset->setObjectName(QStringLiteral("listView_detector_dataset"));

        verticalLayout_20->addWidget(listView_detector_dataset);

        textEdit_detectorOutPath = new QTextEdit(tab);
        textEdit_detectorOutPath->setObjectName(QStringLiteral("textEdit_detectorOutPath"));
        textEdit_detectorOutPath->setGeometry(QRect(670, 530, 401, 21));
        pushButton_detector_output = new QPushButton(tab);
        pushButton_detector_output->setObjectName(QStringLiteral("pushButton_detector_output"));
        pushButton_detector_output->setGeometry(QRect(670, 490, 161, 28));
        verticalLayoutWidget_21 = new QWidget(tab);
        verticalLayoutWidget_21->setObjectName(QStringLiteral("verticalLayoutWidget_21"));
        verticalLayoutWidget_21->setGeometry(QRect(10, 330, 411, 291));
        verticalLayout_21 = new QVBoxLayout(verticalLayoutWidget_21);
        verticalLayout_21->setSpacing(6);
        verticalLayout_21->setContentsMargins(11, 11, 11, 11);
        verticalLayout_21->setObjectName(QStringLiteral("verticalLayout_21"));
        verticalLayout_21->setContentsMargins(0, 0, 0, 0);
        label_23 = new QLabel(verticalLayoutWidget_21);
        label_23->setObjectName(QStringLiteral("label_23"));

        verticalLayout_21->addWidget(label_23);

        listView_detector_weights = new QListView(verticalLayoutWidget_21);
        listView_detector_weights->setObjectName(QStringLiteral("listView_detector_weights"));

        verticalLayout_21->addWidget(listView_detector_weights);

        verticalLayoutWidget_22 = new QWidget(tab);
        verticalLayoutWidget_22->setObjectName(QStringLiteral("verticalLayoutWidget_22"));
        verticalLayoutWidget_22->setGeometry(QRect(460, 330, 281, 131));
        verticalLayout_22 = new QVBoxLayout(verticalLayoutWidget_22);
        verticalLayout_22->setSpacing(6);
        verticalLayout_22->setContentsMargins(11, 11, 11, 11);
        verticalLayout_22->setObjectName(QStringLiteral("verticalLayout_22"));
        verticalLayout_22->setContentsMargins(0, 0, 0, 0);
        label_24 = new QLabel(verticalLayoutWidget_22);
        label_24->setObjectName(QStringLiteral("label_24"));

        verticalLayout_22->addWidget(label_24);

        listView_detector_net_config = new QListView(verticalLayoutWidget_22);
        listView_detector_net_config->setObjectName(QStringLiteral("listView_detector_net_config"));

        verticalLayout_22->addWidget(listView_detector_net_config);

        verticalLayoutWidget_23 = new QWidget(tab);
        verticalLayoutWidget_23->setObjectName(QStringLiteral("verticalLayoutWidget_23"));
        verticalLayoutWidget_23->setGeometry(QRect(470, 490, 186, 131));
        verticalLayout_23 = new QVBoxLayout(verticalLayoutWidget_23);
        verticalLayout_23->setSpacing(6);
        verticalLayout_23->setContentsMargins(11, 11, 11, 11);
        verticalLayout_23->setObjectName(QStringLiteral("verticalLayout_23"));
        verticalLayout_23->setContentsMargins(0, 0, 0, 0);
        label_25 = new QLabel(verticalLayoutWidget_23);
        label_25->setObjectName(QStringLiteral("label_25"));

        verticalLayout_23->addWidget(label_25);

        listView_detector_imp = new QListView(verticalLayoutWidget_23);
        listView_detector_imp->setObjectName(QStringLiteral("listView_detector_imp"));

        verticalLayout_23->addWidget(listView_detector_imp);

        pushButton_detect = new QPushButton(tab);
        pushButton_detect->setObjectName(QStringLiteral("pushButton_detect"));
        pushButton_detect->setGeometry(QRect(1040, 20, 85, 28));
        checkBox_detector_useDepth = new QCheckBox(tab);
        checkBox_detector_useDepth->setObjectName(QStringLiteral("checkBox_detector_useDepth"));
        checkBox_detector_useDepth->setGeometry(QRect(460, 150, 161, 26));
        checkBox_detector_single = new QCheckBox(tab);
        checkBox_detector_single->setObjectName(QStringLiteral("checkBox_detector_single"));
        checkBox_detector_single->setGeometry(QRect(670, 570, 161, 26));
        verticalLayoutWidget_24 = new QWidget(tab);
        verticalLayoutWidget_24->setObjectName(QStringLiteral("verticalLayoutWidget_24"));
        verticalLayoutWidget_24->setGeometry(QRect(760, 330, 191, 131));
        verticalLayout_24 = new QVBoxLayout(verticalLayoutWidget_24);
        verticalLayout_24->setSpacing(6);
        verticalLayout_24->setContentsMargins(11, 11, 11, 11);
        verticalLayout_24->setObjectName(QStringLiteral("verticalLayout_24"));
        verticalLayout_24->setContentsMargins(0, 0, 0, 0);
        label_26 = new QLabel(verticalLayoutWidget_24);
        label_26->setObjectName(QStringLiteral("label_26"));

        verticalLayout_24->addWidget(label_26);

        listView_detector_names_inferencer = new QListView(verticalLayoutWidget_24);
        listView_detector_names_inferencer->setObjectName(QStringLiteral("listView_detector_names_inferencer"));

        verticalLayout_24->addWidget(listView_detector_names_inferencer);

        detector_groupBox_inferencer_params = new QGroupBox(tab);
        detector_groupBox_inferencer_params->setObjectName(QStringLiteral("detector_groupBox_inferencer_params"));
        detector_groupBox_inferencer_params->setGeometry(QRect(810, 150, 321, 191));
        detector_checkBox_use_rgb = new QCheckBox(detector_groupBox_inferencer_params);
        detector_checkBox_use_rgb->setObjectName(QStringLiteral("detector_checkBox_use_rgb"));
        detector_checkBox_use_rgb->setGeometry(QRect(160, 160, 97, 22));
        detector_lineEdit_confidence_thresh = new QLineEdit(detector_groupBox_inferencer_params);
        detector_lineEdit_confidence_thresh->setObjectName(QStringLiteral("detector_lineEdit_confidence_thresh"));
        detector_lineEdit_confidence_thresh->setGeometry(QRect(160, 26, 61, 27));
        label = new QLabel(detector_groupBox_inferencer_params);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(0, 30, 161, 20));
        label_211 = new QLabel(detector_groupBox_inferencer_params);
        label_211->setObjectName(QStringLiteral("label_211"));
        label_211->setGeometry(QRect(0, 60, 141, 17));
        detector_lineEdit_inferencer_scaling_factor = new QLineEdit(detector_groupBox_inferencer_params);
        detector_lineEdit_inferencer_scaling_factor->setObjectName(QStringLiteral("detector_lineEdit_inferencer_scaling_factor"));
        detector_lineEdit_inferencer_scaling_factor->setGeometry(QRect(160, 55, 61, 27));
        detector_lineEdit_mean_sub_blue = new QLineEdit(detector_groupBox_inferencer_params);
        detector_lineEdit_mean_sub_blue->setObjectName(QStringLiteral("detector_lineEdit_mean_sub_blue"));
        detector_lineEdit_mean_sub_blue->setGeometry(QRect(160, 120, 51, 27));
        detector_lineEdit_mean_sub_green = new QLineEdit(detector_groupBox_inferencer_params);
        detector_lineEdit_mean_sub_green->setObjectName(QStringLiteral("detector_lineEdit_mean_sub_green"));
        detector_lineEdit_mean_sub_green->setGeometry(QRect(210, 120, 51, 27));
        detector_lineEdit_mean_sub_red = new QLineEdit(detector_groupBox_inferencer_params);
        detector_lineEdit_mean_sub_red->setObjectName(QStringLiteral("detector_lineEdit_mean_sub_red"));
        detector_lineEdit_mean_sub_red->setGeometry(QRect(260, 120, 51, 27));
        label_33 = new QLabel(detector_groupBox_inferencer_params);
        label_33->setObjectName(QStringLiteral("label_33"));
        label_33->setGeometry(QRect(0, 130, 131, 17));
        label_43 = new QLabel(detector_groupBox_inferencer_params);
        label_43->setObjectName(QStringLiteral("label_43"));
        label_43->setGeometry(QRect(0, 90, 161, 17));
        detector_lineEdit_inferencer_input_width = new QLineEdit(detector_groupBox_inferencer_params);
        detector_lineEdit_inferencer_input_width->setObjectName(QStringLiteral("detector_lineEdit_inferencer_input_width"));
        detector_lineEdit_inferencer_input_width->setGeometry(QRect(160, 88, 61, 27));
        detector_lineEdit_inferencer_input_height = new QLineEdit(detector_groupBox_inferencer_params);
        detector_lineEdit_inferencer_input_height->setObjectName(QStringLiteral("detector_lineEdit_inferencer_input_height"));
        detector_lineEdit_inferencer_input_height->setGeometry(QRect(220, 88, 61, 27));
        tabWidget->addTab(tab, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QStringLiteral("tab_3"));
        verticalLayoutWidget_11 = new QWidget(tab_3);
        verticalLayoutWidget_11->setObjectName(QStringLiteral("verticalLayoutWidget_11"));
        verticalLayoutWidget_11->setGeometry(QRect(440, 10, 191, 131));
        verticalLayout_11 = new QVBoxLayout(verticalLayoutWidget_11);
        verticalLayout_11->setSpacing(6);
        verticalLayout_11->setContentsMargins(11, 11, 11, 11);
        verticalLayout_11->setObjectName(QStringLiteral("verticalLayout_11"));
        verticalLayout_11->setContentsMargins(0, 0, 0, 0);
        label_13 = new QLabel(verticalLayoutWidget_11);
        label_13->setObjectName(QStringLiteral("label_13"));

        verticalLayout_11->addWidget(label_13);

        listView_evaluator_gt_names = new QListView(verticalLayoutWidget_11);
        listView_evaluator_gt_names->setObjectName(QStringLiteral("listView_evaluator_gt_names"));

        verticalLayout_11->addWidget(listView_evaluator_gt_names);

        verticalLayoutWidget_12 = new QWidget(tab_3);
        verticalLayoutWidget_12->setObjectName(QStringLiteral("verticalLayoutWidget_12"));
        verticalLayoutWidget_12->setGeometry(QRect(10, 10, 411, 291));
        verticalLayout_12 = new QVBoxLayout(verticalLayoutWidget_12);
        verticalLayout_12->setSpacing(6);
        verticalLayout_12->setContentsMargins(11, 11, 11, 11);
        verticalLayout_12->setObjectName(QStringLiteral("verticalLayout_12"));
        verticalLayout_12->setContentsMargins(0, 0, 0, 0);
        label_14 = new QLabel(verticalLayoutWidget_12);
        label_14->setObjectName(QStringLiteral("label_14"));

        verticalLayout_12->addWidget(label_14);

        listView_evaluator_gt_dataset = new QListView(verticalLayoutWidget_12);
        listView_evaluator_gt_dataset->setObjectName(QStringLiteral("listView_evaluator_gt_dataset"));

        verticalLayout_12->addWidget(listView_evaluator_gt_dataset);

        verticalLayoutWidget_13 = new QWidget(tab_3);
        verticalLayoutWidget_13->setObjectName(QStringLiteral("verticalLayoutWidget_13"));
        verticalLayoutWidget_13->setGeometry(QRect(440, 150, 220, 131));
        verticalLayout_13 = new QVBoxLayout(verticalLayoutWidget_13);
        verticalLayout_13->setSpacing(6);
        verticalLayout_13->setContentsMargins(11, 11, 11, 11);
        verticalLayout_13->setObjectName(QStringLiteral("verticalLayout_13"));
        verticalLayout_13->setContentsMargins(0, 0, 0, 0);
        label_15 = new QLabel(verticalLayoutWidget_13);
        label_15->setObjectName(QStringLiteral("label_15"));

        verticalLayout_13->addWidget(label_15);

        listView_evaluator_gt_imp = new QListView(verticalLayoutWidget_13);
        listView_evaluator_gt_imp->setObjectName(QStringLiteral("listView_evaluator_gt_imp"));

        verticalLayout_13->addWidget(listView_evaluator_gt_imp);

        verticalLayoutWidget_14 = new QWidget(tab_3);
        verticalLayoutWidget_14->setObjectName(QStringLiteral("verticalLayoutWidget_14"));
        verticalLayoutWidget_14->setGeometry(QRect(450, 330, 191, 131));
        verticalLayout_14 = new QVBoxLayout(verticalLayoutWidget_14);
        verticalLayout_14->setSpacing(6);
        verticalLayout_14->setContentsMargins(11, 11, 11, 11);
        verticalLayout_14->setObjectName(QStringLiteral("verticalLayout_14"));
        verticalLayout_14->setContentsMargins(0, 0, 0, 0);
        label_16 = new QLabel(verticalLayoutWidget_14);
        label_16->setObjectName(QStringLiteral("label_16"));

        verticalLayout_14->addWidget(label_16);

        listView_evaluator_detection_names = new QListView(verticalLayoutWidget_14);
        listView_evaluator_detection_names->setObjectName(QStringLiteral("listView_evaluator_detection_names"));

        verticalLayout_14->addWidget(listView_evaluator_detection_names);

        verticalLayoutWidget_15 = new QWidget(tab_3);
        verticalLayoutWidget_15->setObjectName(QStringLiteral("verticalLayoutWidget_15"));
        verticalLayoutWidget_15->setGeometry(QRect(20, 330, 411, 291));
        verticalLayout_15 = new QVBoxLayout(verticalLayoutWidget_15);
        verticalLayout_15->setSpacing(6);
        verticalLayout_15->setContentsMargins(11, 11, 11, 11);
        verticalLayout_15->setObjectName(QStringLiteral("verticalLayout_15"));
        verticalLayout_15->setContentsMargins(0, 0, 0, 0);
        label_17 = new QLabel(verticalLayoutWidget_15);
        label_17->setObjectName(QStringLiteral("label_17"));

        verticalLayout_15->addWidget(label_17);

        listView_evaluator_dectection_dataset = new QListView(verticalLayoutWidget_15);
        listView_evaluator_dectection_dataset->setObjectName(QStringLiteral("listView_evaluator_dectection_dataset"));

        verticalLayout_15->addWidget(listView_evaluator_dectection_dataset);

        verticalLayoutWidget_16 = new QWidget(tab_3);
        verticalLayoutWidget_16->setObjectName(QStringLiteral("verticalLayoutWidget_16"));
        verticalLayoutWidget_16->setGeometry(QRect(440, 490, 220, 131));
        verticalLayout_16 = new QVBoxLayout(verticalLayoutWidget_16);
        verticalLayout_16->setSpacing(6);
        verticalLayout_16->setContentsMargins(11, 11, 11, 11);
        verticalLayout_16->setObjectName(QStringLiteral("verticalLayout_16"));
        verticalLayout_16->setContentsMargins(0, 0, 0, 0);
        label_18 = new QLabel(verticalLayoutWidget_16);
        label_18->setObjectName(QStringLiteral("label_18"));

        verticalLayout_16->addWidget(label_18);

        listView_evaluator_detection_imp = new QListView(verticalLayoutWidget_16);
        listView_evaluator_detection_imp->setObjectName(QStringLiteral("listView_evaluator_detection_imp"));

        verticalLayout_16->addWidget(listView_evaluator_detection_imp);

        verticalLayoutWidget_17 = new QWidget(tab_3);
        verticalLayoutWidget_17->setObjectName(QStringLiteral("verticalLayoutWidget_17"));
        verticalLayoutWidget_17->setGeometry(QRect(690, 300, 411, 321));
        verticalLayout_17 = new QVBoxLayout(verticalLayoutWidget_17);
        verticalLayout_17->setSpacing(6);
        verticalLayout_17->setContentsMargins(11, 11, 11, 11);
        verticalLayout_17->setObjectName(QStringLiteral("verticalLayout_17"));
        verticalLayout_17->setContentsMargins(0, 0, 0, 0);
        label_19 = new QLabel(verticalLayoutWidget_17);
        label_19->setObjectName(QStringLiteral("label_19"));

        verticalLayout_17->addWidget(label_19);

        listView_evaluator_classFilter = new QListView(verticalLayoutWidget_17);
        listView_evaluator_classFilter->setObjectName(QStringLiteral("listView_evaluator_classFilter"));

        verticalLayout_17->addWidget(listView_evaluator_classFilter);

        pushButton_evaluate = new QPushButton(tab_3);
        pushButton_evaluate->setObjectName(QStringLiteral("pushButton_evaluate"));
        pushButton_evaluate->setGeometry(QRect(1040, 20, 85, 28));
        checkBox_evaluator_merge = new QCheckBox(tab_3);
        checkBox_evaluator_merge->setObjectName(QStringLiteral("checkBox_evaluator_merge"));
        checkBox_evaluator_merge->setGeometry(QRect(710, 30, 191, 26));
        checkBox_evaluator_mix = new QCheckBox(tab_3);
        checkBox_evaluator_mix->setObjectName(QStringLiteral("checkBox_evaluator_mix"));
        checkBox_evaluator_mix->setGeometry(QRect(710, 70, 191, 26));
        checkBox_evaluator_mix->setChecked(true);
        evaluator_ioutype_groupbox = new QGroupBox(tab_3);
        evaluator_ioutype_groupbox->setObjectName(QStringLiteral("evaluator_ioutype_groupbox"));
        evaluator_ioutype_groupbox->setEnabled(true);
        evaluator_ioutype_groupbox->setGeometry(QRect(710, 130, 201, 91));
        QFont font;
        font.setFamily(QStringLiteral("Ubuntu"));
        font.setPointSize(11);
        font.setBold(false);
        font.setUnderline(false);
        font.setWeight(50);
        font.setStrikeOut(false);
        font.setKerning(false);
        evaluator_ioutype_groupbox->setFont(font);
        radioButton_evaluator_iou_bbox = new QRadioButton(evaluator_ioutype_groupbox);
        radioButton_evaluator_iou_bbox->setObjectName(QStringLiteral("radioButton_evaluator_iou_bbox"));
        radioButton_evaluator_iou_bbox->setGeometry(QRect(10, 30, 171, 22));
        radioButton_evaluator_iou_bbox->setChecked(true);
        radioButton_evaluator_iou_seg = new QRadioButton(evaluator_ioutype_groupbox);
        radioButton_evaluator_iou_seg->setObjectName(QStringLiteral("radioButton_evaluator_iou_seg"));
        radioButton_evaluator_iou_seg->setGeometry(QRect(10, 60, 101, 22));
        tabWidget->addTab(tab_3, QString());
        tab_4 = new QWidget();
        tab_4->setObjectName(QStringLiteral("tab_4"));
        verticalLayoutWidget_25 = new QWidget(tab_4);
        verticalLayoutWidget_25->setObjectName(QStringLiteral("verticalLayoutWidget_25"));
        verticalLayoutWidget_25->setGeometry(QRect(630, 520, 186, 131));
        verticalLayout_25 = new QVBoxLayout(verticalLayoutWidget_25);
        verticalLayout_25->setSpacing(6);
        verticalLayout_25->setContentsMargins(11, 11, 11, 11);
        verticalLayout_25->setObjectName(QStringLiteral("verticalLayout_25"));
        verticalLayout_25->setContentsMargins(0, 0, 0, 0);
        label_271 = new QLabel(verticalLayoutWidget_25);
        label_271->setObjectName(QStringLiteral("label_271"));

        verticalLayout_25->addWidget(label_271);

        listView_deploy_impl = new QListView(verticalLayoutWidget_25);
        listView_deploy_impl->setObjectName(QStringLiteral("listView_deploy_impl"));

        verticalLayout_25->addWidget(listView_deploy_impl);

        verticalLayoutWidget_26 = new QWidget(tab_4);
        verticalLayoutWidget_26->setObjectName(QStringLiteral("verticalLayoutWidget_26"));
        verticalLayoutWidget_26->setGeometry(QRect(420, 520, 191, 131));
        verticalLayout_26 = new QVBoxLayout(verticalLayoutWidget_26);
        verticalLayout_26->setSpacing(6);
        verticalLayout_26->setContentsMargins(11, 11, 11, 11);
        verticalLayout_26->setObjectName(QStringLiteral("verticalLayout_26"));
        verticalLayout_26->setContentsMargins(0, 0, 0, 0);
        label_28 = new QLabel(verticalLayoutWidget_26);
        label_28->setObjectName(QStringLiteral("label_28"));

        verticalLayout_26->addWidget(label_28);

        listView_deploy_names_inferencer = new QListView(verticalLayoutWidget_26);
        listView_deploy_names_inferencer->setObjectName(QStringLiteral("listView_deploy_names_inferencer"));

        verticalLayout_26->addWidget(listView_deploy_names_inferencer);

        verticalLayoutWidget_28 = new QWidget(tab_4);
        verticalLayoutWidget_28->setObjectName(QStringLiteral("verticalLayoutWidget_28"));
        verticalLayoutWidget_28->setGeometry(QRect(20, 350, 381, 301));
        verticalLayout_28 = new QVBoxLayout(verticalLayoutWidget_28);
        verticalLayout_28->setSpacing(6);
        verticalLayout_28->setContentsMargins(11, 11, 11, 11);
        verticalLayout_28->setObjectName(QStringLiteral("verticalLayout_28"));
        verticalLayout_28->setContentsMargins(0, 0, 0, 0);
        label_30 = new QLabel(verticalLayoutWidget_28);
        label_30->setObjectName(QStringLiteral("label_30"));

        verticalLayout_28->addWidget(label_30);

        listView_deploy_weights = new QListView(verticalLayoutWidget_28);
        listView_deploy_weights->setObjectName(QStringLiteral("listView_deploy_weights"));

        verticalLayout_28->addWidget(listView_deploy_weights);

        label_311 = new QLabel(tab_4);
        label_311->setObjectName(QStringLiteral("label_311"));
        label_311->setGeometry(QRect(20, 4, 409, 20));
        listView_deploy_input_imp = new QListView(tab_4);
        listView_deploy_input_imp->setObjectName(QStringLiteral("listView_deploy_input_imp"));
        listView_deploy_input_imp->setGeometry(QRect(20, 30, 409, 91));
        deployer_param_groupBox = new QGroupBox(tab_4);
        deployer_param_groupBox->setObjectName(QStringLiteral("deployer_param_groupBox"));
        deployer_param_groupBox->setEnabled(true);
        deployer_param_groupBox->setGeometry(QRect(20, 120, 361, 181));
        deployer_param_groupBox->setFont(font);
        label21 = new QLabel(deployer_param_groupBox);
        label21->setObjectName(QStringLiteral("label21"));
        label21->setGeometry(QRect(20, 60, 61, 31));
        QFont font1;
        font1.setFamily(QStringLiteral("Saab"));
        font1.setPointSize(10);
        font1.setUnderline(false);
        font1.setStrikeOut(false);
        font1.setKerning(true);
        label21->setFont(font1);
        lineEdit_deployer_proxy = new QLineEdit(deployer_param_groupBox);
        lineEdit_deployer_proxy->setObjectName(QStringLiteral("lineEdit_deployer_proxy"));
        lineEdit_deployer_proxy->setGeometry(QRect(90, 60, 211, 23));
        QFont font2;
        font2.setPointSize(11);
        font2.setBold(false);
        font2.setItalic(false);
        font2.setWeight(50);
        lineEdit_deployer_proxy->setFont(font2);
        lineEdit_deployer_format = new QLineEdit(deployer_param_groupBox);
        lineEdit_deployer_format->setObjectName(QStringLiteral("lineEdit_deployer_format"));
        lineEdit_deployer_format->setGeometry(QRect(90, 90, 211, 23));
        lineEdit_deployer_topic = new QLineEdit(deployer_param_groupBox);
        lineEdit_deployer_topic->setObjectName(QStringLiteral("lineEdit_deployer_topic"));
        lineEdit_deployer_topic->setGeometry(QRect(90, 120, 211, 23));
        lineEdit_deployer_name = new QLineEdit(deployer_param_groupBox);
        lineEdit_deployer_name->setObjectName(QStringLiteral("lineEdit_deployer_name"));
        lineEdit_deployer_name->setGeometry(QRect(90, 150, 211, 23));
        label_2 = new QLabel(deployer_param_groupBox);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setEnabled(true);
        label_2->setGeometry(QRect(20, 90, 71, 31));
        QFont font3;
        font3.setPointSize(10);
        label_2->setFont(font3);
        label_3 = new QLabel(deployer_param_groupBox);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(20, 120, 61, 31));
        label_3->setFont(font1);
        label_4 = new QLabel(deployer_param_groupBox);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(20, 150, 61, 21));
        QFont font4;
        font4.setFamily(QStringLiteral("Ubuntu"));
        font4.setPointSize(10);
        font4.setUnderline(false);
        font4.setStrikeOut(false);
        font4.setKerning(true);
        label_4->setFont(font4);
        radioButton_deployer_ros = new QRadioButton(deployer_param_groupBox);
        radioButton_deployer_ros->setObjectName(QStringLiteral("radioButton_deployer_ros"));
        radioButton_deployer_ros->setGeometry(QRect(100, 30, 61, 22));
        radioButton_deployer_ice = new QRadioButton(deployer_param_groupBox);
        radioButton_deployer_ice->setObjectName(QStringLiteral("radioButton_deployer_ice"));
        radioButton_deployer_ice->setGeometry(QRect(170, 30, 117, 22));
        label_51 = new QLabel(deployer_param_groupBox);
        label_51->setObjectName(QStringLiteral("label_51"));
        label_51->setGeometry(QRect(20, 30, 68, 17));
        groupBox_config_option = new QGroupBox(tab_4);
        groupBox_config_option->setObjectName(QStringLiteral("groupBox_config_option"));
        groupBox_config_option->setGeometry(QRect(390, 160, 271, 61));
        deployer_radioButton_manual = new QRadioButton(groupBox_config_option);
        deployer_radioButton_manual->setObjectName(QStringLiteral("deployer_radioButton_manual"));
        deployer_radioButton_manual->setGeometry(QRect(0, 10, 271, 22));
        deployer_radioButton_config = new QRadioButton(groupBox_config_option);
        deployer_radioButton_config->setObjectName(QStringLiteral("deployer_radioButton_config"));
        deployer_radioButton_config->setGeometry(QRect(0, 40, 251, 22));
        textEdit_deployInputPath = new QTextEdit(tab_4);
        textEdit_deployInputPath->setObjectName(QStringLiteral("textEdit_deployInputPath"));
        textEdit_deployInputPath->setGeometry(QRect(730, 60, 401, 21));
        pushButton_deploy_input = new QPushButton(tab_4);
        pushButton_deploy_input->setObjectName(QStringLiteral("pushButton_deploy_input"));
        pushButton_deploy_input->setGeometry(QRect(730, 20, 161, 28));
        pushButton_deploy_process = new QPushButton(tab_4);
        pushButton_deploy_process->setObjectName(QStringLiteral("pushButton_deploy_process"));
        pushButton_deploy_process->setGeometry(QRect(1050, 100, 85, 31));
        pushButton_deploy_process->setCursor(QCursor(Qt::PointingHandCursor));
        pushButton_deploy_process->setAutoDefault(false);
        pushButton_deploy_process->setFlat(false);
        deployer_groupBox_inferencer_params = new QGroupBox(tab_4);
        deployer_groupBox_inferencer_params->setObjectName(QStringLiteral("deployer_groupBox_inferencer_params"));
        deployer_groupBox_inferencer_params->setGeometry(QRect(860, 460, 321, 191));
        deployer_checkBox_use_rgb = new QCheckBox(deployer_groupBox_inferencer_params);
        deployer_checkBox_use_rgb->setObjectName(QStringLiteral("deployer_checkBox_use_rgb"));
        deployer_checkBox_use_rgb->setGeometry(QRect(160, 145, 97, 22));
        label_210 = new QLabel(deployer_groupBox_inferencer_params);
        label_210->setObjectName(QStringLiteral("label_210"));
        label_210->setGeometry(QRect(0, 30, 141, 17));
        deployer_lineEdit_inferencer_scaling_factor = new QLineEdit(deployer_groupBox_inferencer_params);
        deployer_lineEdit_inferencer_scaling_factor->setObjectName(QStringLiteral("deployer_lineEdit_inferencer_scaling_factor"));
        deployer_lineEdit_inferencer_scaling_factor->setGeometry(QRect(160, 27, 61, 27));
        deployer_lineEdit_mean_sub_blue = new QLineEdit(deployer_groupBox_inferencer_params);
        deployer_lineEdit_mean_sub_blue->setObjectName(QStringLiteral("deployer_lineEdit_mean_sub_blue"));
        deployer_lineEdit_mean_sub_blue->setGeometry(QRect(160, 108, 51, 27));
        deployer_lineEdit_mean_sub_green = new QLineEdit(deployer_groupBox_inferencer_params);
        deployer_lineEdit_mean_sub_green->setObjectName(QStringLiteral("deployer_lineEdit_mean_sub_green"));
        deployer_lineEdit_mean_sub_green->setGeometry(QRect(210, 108, 51, 27));
        deployer_lineEdit_mean_sub_red = new QLineEdit(deployer_groupBox_inferencer_params);
        deployer_lineEdit_mean_sub_red->setObjectName(QStringLiteral("deployer_lineEdit_mean_sub_red"));
        deployer_lineEdit_mean_sub_red->setGeometry(QRect(260, 108, 51, 27));
        label_32 = new QLabel(deployer_groupBox_inferencer_params);
        label_32->setObjectName(QStringLiteral("label_32"));
        label_32->setGeometry(QRect(0, 110, 131, 17));
        label_42 = new QLabel(deployer_groupBox_inferencer_params);
        label_42->setObjectName(QStringLiteral("label_42"));
        label_42->setGeometry(QRect(0, 70, 161, 17));
        deployer_lineEdit_inferencer_input_width = new QLineEdit(deployer_groupBox_inferencer_params);
        deployer_lineEdit_inferencer_input_width->setObjectName(QStringLiteral("deployer_lineEdit_inferencer_input_width"));
        deployer_lineEdit_inferencer_input_width->setGeometry(QRect(160, 67, 61, 27));
        deployer_lineEdit_inferencer_input_height = new QLineEdit(deployer_groupBox_inferencer_params);
        deployer_lineEdit_inferencer_input_height->setObjectName(QStringLiteral("deployer_lineEdit_inferencer_input_height"));
        deployer_lineEdit_inferencer_input_height->setGeometry(QRect(220, 67, 61, 27));
        groupbox_deployer_saveOutput = new QGroupBox(tab_4);
        groupbox_deployer_saveOutput->setObjectName(QStringLiteral("groupbox_deployer_saveOutput"));
        groupbox_deployer_saveOutput->setEnabled(false);
        groupbox_deployer_saveOutput->setGeometry(QRect(750, 270, 451, 101));
        pushButton_deployer_output_folder = new QPushButton(groupbox_deployer_saveOutput);
        pushButton_deployer_output_folder->setObjectName(QStringLiteral("pushButton_deployer_output_folder"));
        pushButton_deployer_output_folder->setGeometry(QRect(20, 20, 161, 28));
        textEdit_deployerOutputPath = new QTextEdit(groupbox_deployer_saveOutput);
        textEdit_deployerOutputPath->setObjectName(QStringLiteral("textEdit_deployerOutputPath"));
        textEdit_deployerOutputPath->setGeometry(QRect(20, 60, 401, 21));
        checkBox_deployer_saveOutput = new QCheckBox(tab_4);
        checkBox_deployer_saveOutput->setObjectName(QStringLiteral("checkBox_deployer_saveOutput"));
        checkBox_deployer_saveOutput->setGeometry(QRect(770, 250, 191, 22));
        verticalLayoutWidget_27 = new QWidget(tab_4);
        verticalLayoutWidget_27->setObjectName(QStringLiteral("verticalLayoutWidget_27"));
        verticalLayoutWidget_27->setGeometry(QRect(420, 350, 281, 151));
        verticalLayout_27 = new QVBoxLayout(verticalLayoutWidget_27);
        verticalLayout_27->setSpacing(6);
        verticalLayout_27->setContentsMargins(11, 11, 11, 11);
        verticalLayout_27->setObjectName(QStringLiteral("verticalLayout_27"));
        verticalLayout_27->setContentsMargins(0, 0, 0, 0);
        label_29 = new QLabel(verticalLayoutWidget_27);
        label_29->setObjectName(QStringLiteral("label_29"));

        verticalLayout_27->addWidget(label_29);

        listView_deploy_net_config = new QListView(verticalLayoutWidget_27);
        listView_deploy_net_config->setObjectName(QStringLiteral("listView_deploy_net_config"));

        verticalLayout_27->addWidget(listView_deploy_net_config);

        pushButton_stop_deployer_process = new QPushButton(tab_4);
        pushButton_stop_deployer_process->setObjectName(QStringLiteral("pushButton_stop_deployer_process"));
        pushButton_stop_deployer_process->setGeometry(QRect(1050, 140, 85, 31));
        deployer_conf_horizontalSlider = new QSlider(tab_4);
        deployer_conf_horizontalSlider->setObjectName(QStringLiteral("deployer_conf_horizontalSlider"));
        deployer_conf_horizontalSlider->setGeometry(QRect(450, 300, 160, 16));
        deployer_conf_horizontalSlider->setStyleSheet(QStringLiteral(""));
        deployer_conf_horizontalSlider->setMaximum(100);
        deployer_conf_horizontalSlider->setSliderPosition(20);
        deployer_conf_horizontalSlider->setTracking(true);
        deployer_conf_horizontalSlider->setOrientation(Qt::Horizontal);
        deployer_conf_horizontalSlider->setTickPosition(QSlider::TicksBothSides);
        deployer_confidence_label = new QLabel(tab_4);
        deployer_confidence_label->setObjectName(QStringLiteral("deployer_confidence_label"));
        deployer_confidence_label->setGeometry(QRect(447, 270, 171, 20));
        deployer_confidence_lineEdit = new QLineEdit(tab_4);
        deployer_confidence_lineEdit->setObjectName(QStringLiteral("deployer_confidence_lineEdit"));
        deployer_confidence_lineEdit->setGeometry(QRect(608, 267, 41, 27));
        deployer_cameraID_groupBox = new QGroupBox(tab_4);
        deployer_cameraID_groupBox->setObjectName(QStringLiteral("deployer_cameraID_groupBox"));
        deployer_cameraID_groupBox->setEnabled(true);
        deployer_cameraID_groupBox->setGeometry(QRect(460, 30, 101, 71));
        deployer_camera_spinBox = new QSpinBox(deployer_cameraID_groupBox);
        deployer_camera_spinBox->setObjectName(QStringLiteral("deployer_camera_spinBox"));
        deployer_camera_spinBox->setEnabled(true);
        deployer_camera_spinBox->setGeometry(QRect(0, 30, 57, 27));
        deployer_camera_spinBox->setMinimum(-1);
        tabWidget->addTab(tab_4, QString());
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1225, 25));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(4);
        pushButton_deploy_process->setDefault(false);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0));
        pushButton->setText(QApplication::translate("MainWindow", "View", 0));
        label_1111_->setText(QApplication::translate("MainWindow", "Input Dataset", 0));
        label_27->setText(QApplication::translate("MainWindow", "Dataset Names", 0));
        label_31->setText(QApplication::translate("MainWindow", "Dataset implementation", 0));
        label_41->setText(QApplication::translate("MainWindow", "Filter by class", 0));
        checkBox_evaluator_show_depth->setText(QApplication::translate("MainWindow", "Show Depth Images", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_1), QApplication::translate("MainWindow", "Viewer", 0));
        label_5->setText(QApplication::translate("MainWindow", "Reader Dataset implementation", 0));
        label_6->setText(QApplication::translate("MainWindow", "Dataset Names", 0));
        label_7->setText(QApplication::translate("MainWindow", "Input Dataset", 0));
        label_8->setText(QApplication::translate("MainWindow", "Filter by class", 0));
        label_9->setText(QApplication::translate("MainWindow", "Writer Dataset Implementation", 0));
        label_101->setText(QApplication::translate("MainWindow", "Writer Dataset Names", 0));
        checkBox_use_writernames->setText(QApplication::translate("MainWindow", "Map To Writer Names, May Lead to data loss", 0));
        label_11->setText(QApplication::translate("MainWindow", "Output Path", 0));
        pushButton_converter_output->setText(QApplication::translate("MainWindow", "Select Folder", 0));
        pushButton_convert->setText(QApplication::translate("MainWindow", "Convert", 0));
        checkBox_splitActive->setText(QApplication::translate("MainWindow", "Split into test and train", 0));
        label_10->setText(QApplication::translate("MainWindow", "Train Ratio", 0));
        label_12->setText(QApplication::translate("MainWindow", "Writer Configuration", 0));
        checkBox_converter_write_images->setText(QApplication::translate("MainWindow", "Write Images", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("MainWindow", "Converter", 0));
        label_20->setText(QApplication::translate("MainWindow", "Dataset implementation", 0));
        label_21->setText(QApplication::translate("MainWindow", "Dataset Names", 0));
        label_22->setText(QApplication::translate("MainWindow", "Input Dataset", 0));
        pushButton_detector_output->setText(QApplication::translate("MainWindow", "Select Output Folder", 0));
        label_23->setText(QApplication::translate("MainWindow", "Net weights", 0));
        label_24->setText(QApplication::translate("MainWindow", "Net Configuration", 0));
        label_25->setText(QApplication::translate("MainWindow", "Inferencer Implementation", 0));
        pushButton_detect->setText(QApplication::translate("MainWindow", "Detect", 0));
        checkBox_detector_useDepth->setText(QApplication::translate("MainWindow", "Use depth images", 0));
        checkBox_detector_single->setText(QApplication::translate("MainWindow", "Single Evaluation", 0));
        label_26->setText(QApplication::translate("MainWindow", "Inferencer names", 0));
        detector_groupBox_inferencer_params->setTitle(QApplication::translate("MainWindow", "Inferencer Parameters:", 0));
        detector_checkBox_use_rgb->setText(QApplication::translate("MainWindow", "Use RGB", 0));
        label->setText(QApplication::translate("MainWindow", "Confidence Threshold:", 0));
        label_211->setText(QApplication::translate("MainWindow", "Scaling Factor:", 0));
        detector_lineEdit_mean_sub_blue->setPlaceholderText(QApplication::translate("MainWindow", "B", 0));
        detector_lineEdit_mean_sub_green->setPlaceholderText(QApplication::translate("MainWindow", "G", 0));
        detector_lineEdit_mean_sub_red->setPlaceholderText(QApplication::translate("MainWindow", "R", 0));
        label_33->setText(QApplication::translate("MainWindow", "Mean Subtraction:", 0));
        label_43->setText(QApplication::translate("MainWindow", "Inferencer Input Size:", 0));
        detector_lineEdit_inferencer_input_width->setInputMask(QString());
        detector_lineEdit_inferencer_input_width->setPlaceholderText(QApplication::translate("MainWindow", "WIdth", 0));
        detector_lineEdit_inferencer_input_height->setPlaceholderText(QApplication::translate("MainWindow", "Height", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("MainWindow", "Detector", 0));
        label_13->setText(QApplication::translate("MainWindow", "Dataset Names", 0));
        label_14->setText(QApplication::translate("MainWindow", "Input Grond Thruth Dataset", 0));
        label_15->setText(QApplication::translate("MainWindow", "Reader Dataset implementation", 0));
        label_16->setText(QApplication::translate("MainWindow", "Dataset Names", 0));
        label_17->setText(QApplication::translate("MainWindow", "Input Detection Dataset", 0));
        label_18->setText(QApplication::translate("MainWindow", "Reader Dataset implementation", 0));
        label_19->setText(QApplication::translate("MainWindow", "Filter by class", 0));
        pushButton_evaluate->setText(QApplication::translate("MainWindow", "Evaluate", 0));
        checkBox_evaluator_merge->setText(QApplication::translate("MainWindow", "Merge all person clases", 0));
        checkBox_evaluator_mix->setText(QApplication::translate("MainWindow", "Add mix evaluation", 0));
#ifndef QT_NO_WHATSTHIS
        evaluator_ioutype_groupbox->setWhatsThis(QString());
#endif // QT_NO_WHATSTHIS
        evaluator_ioutype_groupbox->setTitle(QApplication::translate("MainWindow", "IOU Type", 0));
        radioButton_evaluator_iou_bbox->setText(QApplication::translate("MainWindow", "Use Bounding Boxes", 0));
        radioButton_evaluator_iou_seg->setText(QApplication::translate("MainWindow", "Use Masks", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QApplication::translate("MainWindow", "Evaluator", 0));
        label_271->setText(QApplication::translate("MainWindow", "Inferencer Implementation", 0));
        label_28->setText(QApplication::translate("MainWindow", "Inferencer names", 0));
        label_30->setText(QApplication::translate("MainWindow", "Net weights", 0));
#ifndef QT_NO_WHATSTHIS
        label_311->setWhatsThis(QString());
#endif // QT_NO_WHATSTHIS
        label_311->setText(QApplication::translate("MainWindow", "Deployer Input Type", 0));
#ifndef QT_NO_WHATSTHIS
        listView_deploy_input_imp->setWhatsThis(QString());
#endif // QT_NO_WHATSTHIS
#ifndef QT_NO_WHATSTHIS
        deployer_param_groupBox->setWhatsThis(QString());
#endif // QT_NO_WHATSTHIS
        deployer_param_groupBox->setTitle(QApplication::translate("MainWindow", "Camera Stream Parameters:", 0));
        label21->setText(QApplication::translate("MainWindow", "Proxy:", 0));
        lineEdit_deployer_proxy->setText(QApplication::translate("MainWindow", "cam1:tcp -h localhost -p 9999", 0));
        lineEdit_deployer_format->setText(QApplication::translate("MainWindow", "RGB8", 0));
        lineEdit_deployer_topic->setText(QApplication::translate("MainWindow", "DetectionSuite/Deployer", 0));
        lineEdit_deployer_name->setText(QApplication::translate("MainWindow", "cam1", 0));
        label_2->setText(QApplication::translate("MainWindow", "Format:", 0));
        label_3->setText(QApplication::translate("MainWindow", "Topic:", 0));
        label_4->setText(QApplication::translate("MainWindow", "Name:", 0));
        radioButton_deployer_ros->setText(QApplication::translate("MainWindow", "ROS", 0));
        radioButton_deployer_ice->setText(QApplication::translate("MainWindow", "ICE", 0));
        label_51->setText(QApplication::translate("MainWindow", "Server:", 0));
#ifndef QT_NO_WHATSTHIS
        groupBox_config_option->setWhatsThis(QString());
#endif // QT_NO_WHATSTHIS
        groupBox_config_option->setTitle(QString());
        deployer_radioButton_manual->setText(QApplication::translate("MainWindow", "Enter Config Parameters Manually", 0));
        deployer_radioButton_config->setText(QApplication::translate("MainWindow", "Select Config File", 0));
#ifndef QT_NO_WHATSTHIS
        textEdit_deployInputPath->setWhatsThis(QString());
#endif // QT_NO_WHATSTHIS
#ifndef QT_NO_WHATSTHIS
        pushButton_deploy_input->setWhatsThis(QString());
#endif // QT_NO_WHATSTHIS
        pushButton_deploy_input->setText(QApplication::translate("MainWindow", "Select Input", 0));
#ifndef QT_NO_TOOLTIP
        pushButton_deploy_process->setToolTip(QString());
#endif // QT_NO_TOOLTIP
        pushButton_deploy_process->setText(QApplication::translate("MainWindow", "Process", 0));
        deployer_groupBox_inferencer_params->setTitle(QApplication::translate("MainWindow", "Inferencer Parameters", 0));
        deployer_checkBox_use_rgb->setText(QApplication::translate("MainWindow", "Use RGB", 0));
        label_210->setText(QApplication::translate("MainWindow", "Scaling Factor:", 0));
        deployer_lineEdit_mean_sub_blue->setPlaceholderText(QApplication::translate("MainWindow", "B", 0));
        deployer_lineEdit_mean_sub_green->setPlaceholderText(QApplication::translate("MainWindow", "G", 0));
        deployer_lineEdit_mean_sub_red->setPlaceholderText(QApplication::translate("MainWindow", "R", 0));
        label_32->setText(QApplication::translate("MainWindow", "Mean Subtraction:", 0));
        label_42->setText(QApplication::translate("MainWindow", "Inferencer Input Size:", 0));
        deployer_lineEdit_inferencer_input_width->setInputMask(QString());
        deployer_lineEdit_inferencer_input_width->setPlaceholderText(QApplication::translate("MainWindow", "Width", 0));
        deployer_lineEdit_inferencer_input_height->setPlaceholderText(QApplication::translate("MainWindow", "Height", 0));
        groupbox_deployer_saveOutput->setTitle(QString());
#ifndef QT_NO_WHATSTHIS
        pushButton_deployer_output_folder->setWhatsThis(QString());
#endif // QT_NO_WHATSTHIS
        pushButton_deployer_output_folder->setText(QApplication::translate("MainWindow", "Select Output Folder", 0));
#ifndef QT_NO_WHATSTHIS
        checkBox_deployer_saveOutput->setWhatsThis(QString());
#endif // QT_NO_WHATSTHIS
        checkBox_deployer_saveOutput->setText(QApplication::translate("MainWindow", "Save Output Inferences", 0));
        label_29->setText(QApplication::translate("MainWindow", "Net Configuration", 0));
        pushButton_stop_deployer_process->setText(QApplication::translate("MainWindow", "Stop", 0));
#ifndef QT_NO_TOOLTIP
        deployer_conf_horizontalSlider->setToolTip(QApplication::translate("MainWindow", "<html><head/><body><p>Confidence Threshold for inferencer, typically 0.2</p></body></html>", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_WHATSTHIS
        deployer_conf_horizontalSlider->setWhatsThis(QApplication::translate("MainWindow", "<html><head/><body><p>Confidence Threshold for inferencer, typically 0.2</p></body></html>", 0));
#endif // QT_NO_WHATSTHIS
        deployer_confidence_label->setText(QApplication::translate("MainWindow", "Confidence Threshold: ", 0));
        deployer_confidence_lineEdit->setInputMask(QString());
        deployer_confidence_lineEdit->setText(QApplication::translate("MainWindow", "0.2", 0));
        deployer_confidence_lineEdit->setPlaceholderText(QString());
#ifndef QT_NO_TOOLTIP
        deployer_cameraID_groupBox->setToolTip(QApplication::translate("MainWindow", "Camera ID starting form 0, -1 for any camera vaiable", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_WHATSTHIS
        deployer_cameraID_groupBox->setWhatsThis(QApplication::translate("MainWindow", "Camera ID starting form 0, -1 for any camera vaiable", 0));
#endif // QT_NO_WHATSTHIS
        deployer_cameraID_groupBox->setTitle(QApplication::translate("MainWindow", "Camera ID:", 0));
#ifndef QT_NO_TOOLTIP
        deployer_camera_spinBox->setToolTip(QApplication::translate("MainWindow", "Camera ID starting form 0, -1 for any camera vaiable", 0));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_WHATSTHIS
        deployer_camera_spinBox->setWhatsThis(QApplication::translate("MainWindow", "Camera ID starting form 0, -1 for any camera vaiable", 0));
#endif // QT_NO_WHATSTHIS
        deployer_camera_spinBox->setPrefix(QString());
        tabWidget->setTabText(tabWidget->indexOf(tab_4), QApplication::translate("MainWindow", "Deploy", 0));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
