#include "qttest.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	qtTest w;
	w.show();
	return a.exec();
}
