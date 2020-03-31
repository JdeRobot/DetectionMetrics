/****************************************************************************
** Meta object code from reading C++ file 'appconfig.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.9.5)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../DatasetEvaluationApp/gui/appconfig.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'appconfig.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.9.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_appconfig_t {
    QByteArrayData data[9];
    char stringdata0[158];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_appconfig_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_appconfig_t qt_meta_stringdata_appconfig = {
    {
QT_MOC_LITERAL(0, 0, 9), // "appconfig"
QT_MOC_LITERAL(1, 10, 23), // "handleToolbuttonWeights"
QT_MOC_LITERAL(2, 34, 0), // ""
QT_MOC_LITERAL(3, 35, 21), // "handleToolbuttonNames"
QT_MOC_LITERAL(4, 57, 19), // "handleToolbuttonCfg"
QT_MOC_LITERAL(5, 77, 25), // "handleToolbuttonAppconfig"
QT_MOC_LITERAL(6, 103, 20), // "handleToolbuttonEval"
QT_MOC_LITERAL(7, 124, 14), // "handleCheckbox"
QT_MOC_LITERAL(8, 139, 18) // "handlePushbuttonOK"

    },
    "appconfig\0handleToolbuttonWeights\0\0"
    "handleToolbuttonNames\0handleToolbuttonCfg\0"
    "handleToolbuttonAppconfig\0"
    "handleToolbuttonEval\0handleCheckbox\0"
    "handlePushbuttonOK"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_appconfig[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   49,    2, 0x08 /* Private */,
       3,    0,   50,    2, 0x08 /* Private */,
       4,    0,   51,    2, 0x08 /* Private */,
       5,    0,   52,    2, 0x08 /* Private */,
       6,    0,   53,    2, 0x08 /* Private */,
       7,    0,   54,    2, 0x08 /* Private */,
       8,    0,   55,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void appconfig::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        appconfig *_t = static_cast<appconfig *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->handleToolbuttonWeights(); break;
        case 1: _t->handleToolbuttonNames(); break;
        case 2: _t->handleToolbuttonCfg(); break;
        case 3: _t->handleToolbuttonAppconfig(); break;
        case 4: _t->handleToolbuttonEval(); break;
        case 5: _t->handleCheckbox(); break;
        case 6: _t->handlePushbuttonOK(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObject appconfig::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_appconfig.data,
      qt_meta_data_appconfig,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *appconfig::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *appconfig::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_appconfig.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int appconfig::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
