/********************************************************************************
** Form generated from reading UI file 'qttest.ui'
**
** Created by: Qt User Interface Compiler version 5.4.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QTTEST_H
#define UI_QTTEST_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>
#include "myglwidget.h"

QT_BEGIN_NAMESPACE

class Ui_qtTestClass
{
public:
    QAction *actionLoad_PLY;
    QAction *actionSave_Config;
    QAction *actionLoad_Config;
    QAction *actionExit;
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    MyGLWidget *glwidget;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *qtTestClass)
    {
        if (qtTestClass->objectName().isEmpty())
            qtTestClass->setObjectName(QStringLiteral("qtTestClass"));
        qtTestClass->resize(728, 514);
        actionLoad_PLY = new QAction(qtTestClass);
        actionLoad_PLY->setObjectName(QStringLiteral("actionLoad_PLY"));
        actionSave_Config = new QAction(qtTestClass);
        actionSave_Config->setObjectName(QStringLiteral("actionSave_Config"));
        actionLoad_Config = new QAction(qtTestClass);
        actionLoad_Config->setObjectName(QStringLiteral("actionLoad_Config"));
        actionExit = new QAction(qtTestClass);
        actionExit->setObjectName(QStringLiteral("actionExit"));
        centralWidget = new QWidget(qtTestClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        glwidget = new MyGLWidget(centralWidget);
        glwidget->setObjectName(QStringLiteral("glwidget"));

        gridLayout->addWidget(glwidget, 0, 0, 1, 1);

        qtTestClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(qtTestClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 728, 21));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        qtTestClass->setMenuBar(menuBar);
        statusBar = new QStatusBar(qtTestClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(statusBar->sizePolicy().hasHeightForWidth());
        statusBar->setSizePolicy(sizePolicy);
        qtTestClass->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuFile->addAction(actionLoad_PLY);
        menuFile->addAction(actionSave_Config);
        menuFile->addAction(actionLoad_Config);
        menuFile->addSeparator();
        menuFile->addAction(actionExit);

        retranslateUi(qtTestClass);

        QMetaObject::connectSlotsByName(qtTestClass);
    } // setupUi

    void retranslateUi(QMainWindow *qtTestClass)
    {
        qtTestClass->setWindowTitle(QApplication::translate("qtTestClass", "qtTest", 0));
        actionLoad_PLY->setText(QApplication::translate("qtTestClass", "Load PLY", 0));
        actionSave_Config->setText(QApplication::translate("qtTestClass", "Save Config", 0));
        actionLoad_Config->setText(QApplication::translate("qtTestClass", "Load Config", 0));
        actionExit->setText(QApplication::translate("qtTestClass", "Exit", 0));
        menuFile->setTitle(QApplication::translate("qtTestClass", "File", 0));
    } // retranslateUi

};

namespace Ui {
    class qtTestClass: public Ui_qtTestClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTTEST_H
