from typing import Callable, Dict, Iterable, List, Optional, Type, Union

from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDockWidget,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QLabel,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QSpacerItem,
    QSizePolicy
)
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class ParameterDock(QDockWidget):
    def __init__(self, name: str, 
                 main_window: QMainWindow,
                 parameters):
        super().__init__(name)
        self.name = name
        self.main_window = main_window

        self.setObjectName(self.name + "Dock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        dock_widget = QWidget()
        dock_widget.setObjectName(self.name + "Widget")

        layout = QVBoxLayout()
        parameterTreeWidget = ParameterTree(dock_widget)
        parameterTreeWidget.setObjectName("ParameterTree")
        layout.addWidget(parameterTreeWidget)

        self.parameters = Parameter.create(name='params', type='group',
                                           children=parameters)
        parameterTreeWidget.setParameters(self.parameters, showTop=False)

        horiz = QHBoxLayout()
        spacerItem1 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        horiz.addItem(spacerItem1)
        loadParametersButton = QPushButton(dock_widget)
        loadParametersButton.setObjectName("loadParametersButton")
        horiz.addWidget(loadParametersButton)
        saveParametersButton = QPushButton(dock_widget)
        saveParametersButton.setObjectName("saveParametersButton")
        horiz.addWidget(saveParametersButton)
        layout.addLayout(horiz)

        dock_widget.setLayout(layout)
        self.setWidget(dock_widget)

        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self)
        self.main_window.viewMenu.addAction(self.toggleViewAction())
        

