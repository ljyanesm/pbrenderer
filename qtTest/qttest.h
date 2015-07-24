#ifndef QTTEST_H
#define QTTEST_H

#include <QtWidgets/QMainWindow>
#include "ui_qttest.h"

class qtTest : public QMainWindow
{
	Q_OBJECT

public:
	qtTest(QWidget *parent = 0);
	~qtTest();

private:
	Ui::qtTestClass ui;
};

#endif // QTTEST_H
