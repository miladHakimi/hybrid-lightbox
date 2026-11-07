from __future__ import annotations
import copy
import math
import sys
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, Signal, QSize, QRect, QPoint, QPointF, QObject, QThread, Slot, QStandardPaths, QTimer, QItemSelectionModel, QEvent
from PySide6.QtGui import QAction, QColor, QCursor, QDrag, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFrame,
 QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
 QSizePolicy, QSlider, QSpinBox, QSplitter, QStackedWidget, QStatusBar, QStyledItemDelegate, QVBoxLayout, QWidget, QFormLayout, QMenu, QRubberBand, QCheckBox, QScrollArea, QLayout, QKeySequenceEdit)
from PySide6.QtWidgets import QInputDialog
from .cli import DEFAULT_STL_DIMENSIONS_MM, export
from .model import LayerMode, Project, magic_wand_region
from .optics import OPTIMIZER_NOZZLE_DIAMETER_MM, optimize_rear_openings, preview_u8
from .view import ConnectorCache, automatic_view, brush_cursor_diameter, connector_view_preview, image_view_preview, mask_outline, source_image_preview, supports_drag_paint, valid_segment_indices, view_for_tool
from .keybindings import COMMANDS, DEFAULT_BINDINGS, KeybindingStore, dispatch_keybinding

COLORS = ["#ff6b6b", "#f6bd60", "#59c3c3", "#7aa2ff", "#c77dff"]
LAYER_PREVIEW_COLORS = ["#17151f", "#3f2e68", "#563d87", "#6c50a3", "#8669bb"]
STYLE = """
QMainWindow, QWidget#root { background:#111318; color:#e9edf5; }
QWidget { font-family:-apple-system, BlinkMacSystemFont, "Inter", sans-serif; font-size:13px; color:#d8dee9; }
QFrame#topbar { background:#171a21; border-bottom:1px solid #292e39; }
QFrame#sidebar, QFrame#inspector { background:#171a21; border:1px solid #292e39; border-radius:14px; }
QScrollArea#sidebarScroll, QScrollArea#sidebarScroll > QWidget > QWidget { background:transparent; border:none; }
QFrame#stage { background:#0c0e12; border:1px solid #292e39; border-radius:16px; }
QFrame#toolstrip { background:#171a21; border:1px solid #303642; border-radius:10px; }
QLabel#brand { font-size:19px; font-weight:700; color:white; }
QLabel#eyebrow { color:#7d8799; font-size:11px; font-weight:700; }
QLabel#heading { color:#f5f7fb; font-size:15px; font-weight:700; }
QLabel#muted { color:#8e98aa; }
QLabel#emptyTitle { color:white; font-size:21px; font-weight:700; }
QLabel#emptyCopy { color:#8993a5; font-size:14px; }
QPushButton { background:#252a35; color:#eef2f8; border:1px solid #363d4a; border-radius:8px; padding:8px 12px; font-weight:600; }
QPushButton:hover { background:#303745; border-color:#485162; }
QPushButton#primary { background:#7c5cff; border-color:#8d72ff; color:white; }
QPushButton#primary:hover { background:#8b6cff; }
QPushButton#ghost { background:transparent; border-color:#303642; color:#aeb7c6; }
QSlider::groove:horizontal { height:5px; background:#303642; border-radius:2px; }
QSlider::sub-page:horizontal { background:#7c5cff; border-radius:2px; }
QSlider::handle:horizontal { width:15px; margin:-5px 0; background:#e9edf5; border:1px solid #8d72ff; border-radius:7px; }
QListWidget { background:transparent; border:none; outline:none; padding:2px; }
QListWidget::item { background:#1e222b; border:1px solid #2d333f; border-radius:9px; padding:10px; margin:3px 0; }
QListWidget::item:hover { background:#252a35; }
QListWidget::item:selected { background:#302a4c; border-color:#745bf2; color:white; }
QFrame#layerDock { background:#171a21; border:1px solid #303642; border-radius:10px; }
QSplitter#previewSplitter::handle { background:#303642; border-radius:3px; margin:2px; }
QSplitter#previewSplitter::handle:hover { background:#745bf2; }
QListWidget#layerStrip { background:transparent; border:none; padding:2px; }
QListWidget#layerStrip::item { margin:3px; padding:5px; }
QComboBox, QDoubleSpinBox, QSpinBox { background:#20242d; color:#ecf0f7; border:1px solid #343b48; border-radius:7px; padding:7px 9px; min-height:18px; }
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover { border-color:#566175; }
QComboBox::drop-down { border:none; width:22px; }
QComboBox QAbstractItemView { background:#20242d; color:white; selection-background-color:#745bf2; border:1px solid #3b4351; }
QMenu { background:#20242d; color:#eef2f8; border:1px solid #3b4351; border-radius:8px; padding:5px; }
QMenu::item { padding:8px 24px 8px 10px; border-radius:5px; }
QMenu::item:selected { background:#745bf2; color:white; }
QMenu::item:disabled { color:#8e98aa; font-weight:600; }
QMenu::separator { height:1px; background:#343b48; margin:4px 7px; }
QStatusBar { background:#111318; color:#7f899b; border-top:1px solid #232833; }
"""

def array_pixmap(array):
    data=np.ascontiguousarray(array)
    fmt=QImage.Format_Grayscale8 if data.ndim==2 else QImage.Format_RGB888
    image=QImage(data.data,data.shape[1],data.shape[0],data.strides[0],fmt)
    return QPixmap.fromImage(image.copy())

def dot_icon(color):
    pm=QPixmap(18,18); pm.fill(Qt.transparent); p=QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color)); p.setPen(Qt.NoPen); p.drawEllipse(3,3,12,12); p.end(); return QIcon(pm)

def checkerboard(shape,cell=None):
    """Create the familiar transparency grid used by layer editors."""
    height,width=shape;cell=cell or max(4,min(height,width)//12)
    yy,xx=np.indices((height,width));light=((xx//cell+yy//cell)%2)==0
    result=np.empty((height,width,3),np.uint8);result[light]=[232,234,238];result[~light]=[190,194,201]
    return result

def layer_preview(mask,color):
    result=checkerboard(mask.shape);result[np.asarray(mask,bool)]=QColor(color).getRgb()[:3]
    return result

class PreviewLabel(QLabel):
    """A pixmap label whose image follows the space assigned by its layout."""
    def __init__(self):
        super().__init__();self.source_pixmap=QPixmap();self.setAlignment(Qt.AlignCenter);self.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored)
    def set_source_pixmap(self,pixmap):
        self.source_pixmap=pixmap;self._rescale()
    def clear(self):
        self.source_pixmap=QPixmap();super().clear()
    def resizeEvent(self,event):
        super().resizeEvent(event);self._rescale()
    def _rescale(self):
        if self.source_pixmap.isNull():return
        size=self.size()-QSize(14,14)
        if size.width()>0 and size.height()>0:super().setPixmap(self.source_pixmap.scaled(size,Qt.KeepAspectRatio,Qt.SmoothTransformation))

class LayerItemDelegate(QStyledItemDelegate):
    """Draw the compact remove affordance without obscuring layer previews."""
    def paint(self,painter,option,index):
        super().paint(painter,option,index)
        if index.data(Qt.UserRole)=="add":return
        circle=QRect(option.rect.right()-25,option.rect.top()+7,18,18)
        painter.save();painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor("#303642"));painter.setPen(QPen(QColor("#aeb7c6"),1));painter.drawEllipse(circle)
        painter.setPen(QPen(QColor("#eef2f8"),1.5));center=circle.center();painter.drawLine(center.x()-3,center.y()-3,center.x()+3,center.y()+3);painter.drawLine(center.x()+3,center.y()-3,center.x()-3,center.y()+3);painter.restore()
        if index.data(Qt.UserRole+2):
            linked=bool(index.data(Qt.UserRole+1));color=QColor("#48d17f" if linked else "#687284");cy=option.rect.center().y();cx=option.rect.right()-11
            painter.save();painter.setRenderHint(QPainter.Antialiasing);painter.setPen(QPen(color,2.5));painter.setBrush(QColor("#171a21"));painter.drawEllipse(QPoint(cx-4,cy),5,5);painter.drawEllipse(QPoint(cx+4,cy),5,5);painter.drawLine(cx-2,cy,cx+2,cy);painter.restore()

class LayerList(QListWidget):
    add_requested=Signal();remove_requested=Signal(int);move_requested=Signal(int,int);link_toggled=Signal(int)
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.dragged_row=None;self.drop_insertion=None
    def mousePressEvent(self,event):
        item=self.itemAt(event.position().toPoint())
        if item and item.data(Qt.UserRole)=="add":
            self.add_requested.emit();event.accept();return
        if item:
            rect=self.visualItemRect(item);remove_rect=QRect(rect.right()-29,rect.top()+3,26,26)
            link_rect=QRect(rect.right()-16,rect.center().y()-16,32,32)
            if item.data(Qt.UserRole+2) and link_rect.contains(event.position().toPoint()):
                self.link_toggled.emit(self.row(item));event.accept();return
            if remove_rect.contains(event.position().toPoint()):
                self.remove_requested.emit(self.row(item));event.accept();return
        super().mousePressEvent(event)
    def startDrag(self,supported_actions):
        item=self.currentItem()
        if item is None or item.data(Qt.UserRole)=="add":return
        self.dragged_row=self.row(item)
        drag=QDrag(self);drag.setMimeData(self.model().mimeData([self.currentIndex()]))
        preview=item.icon().pixmap(self.iconSize())
        if not preview.isNull():drag.setPixmap(preview);drag.setHotSpot(QPoint(preview.width()//2,preview.height()//2))
        try:drag.exec(Qt.MoveAction)
        finally:
            self.dragged_row=None;self.drop_insertion=None;self.viewport().update()
    def dragEnterEvent(self,event):
        if event.source() is self and self.dragged_row is not None:event.acceptProposedAction()
        else:event.ignore()
    def dragMoveEvent(self,event):
        if event.source() is not self or self.dragged_row is None:
            event.ignore();return
        self.drop_insertion=self._insertion_at(event.position().toPoint())
        self.viewport().update();event.acceptProposedAction()
    def dragLeaveEvent(self,event):
        self.drop_insertion=None;self.viewport().update();super().dragLeaveEvent(event)
    def _layer_count(self):
        return self.count()-int(self.count()>0 and self.item(self.count()-1).data(Qt.UserRole)=="add")
    def _insertion_at(self,position):
        for row in range(self._layer_count()):
            if position.x()<self.visualItemRect(self.item(row)).center().x():return row
        return self._layer_count()
    def paintEvent(self,event):
        super().paintEvent(event)
        if self.drop_insertion is None or not self._layer_count():return
        insertion=self.drop_insertion
        if insertion==0:x=self.visualItemRect(self.item(0)).left()-3
        elif insertion==self._layer_count():x=self.visualItemRect(self.item(insertion-1)).right()+3
        else:
            before=self.visualItemRect(self.item(insertion-1));after=self.visualItemRect(self.item(insertion))
            x=(before.right()+after.left())//2
        first=self.visualItemRect(self.item(0));last=self.visualItemRect(self.item(self._layer_count()-1))
        painter=QPainter(self.viewport());painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#9b82ff"),4,Qt.SolidLine,Qt.RoundCap));painter.drawLine(x,first.top()+8,x,last.bottom()-8);painter.end()
    def dropEvent(self,event):
        if event.source() is not self or self.dragged_row is None:
            event.ignore();return
        source=self.dragged_row
        insertion=self.drop_insertion if self.drop_insertion is not None else self._insertion_at(event.position().toPoint())
        destination=insertion-(source<insertion)
        destination=max(0,min(destination,self._layer_count()-1))
        self.drop_insertion=None;self.viewport().update()
        if destination!=source:self.move_requested.emit(source,destination)
        event.acceptProposedAction()

class Canvas(QLabel):
    painted=Signal(int,int,bool)
    edit_started=Signal()
    hovered=Signal(int,int); hover_left=Signal()
    segment_requested=Signal(int,int,str)
    segment_picker_requested=Signal(int,int,object)
    wand_requested=Signal(int,int,bool)
    crop_requested=Signal(int,int,int,int)
    transform_dragged=Signal(str,float,float)
    def __init__(self):
        super().__init__(); self.crop_mode=False;self.tool_mode="brush";self.wand_inverse=False;self.crop_start=None;self.rubber_band=QRubberBand(QRubberBand.Rectangle,self);self.setAlignment(Qt.AlignCenter); self.setMinimumSize(460,300); self.setMouseTracking(True); self.setStyleSheet("background:#0c0e12;border-radius:14px;");self.setToolTip("Pinch to zoom · Double-click to reset zoom · Long-press to choose overlapping segments")
        # Ignore the pixmap's size hint so a growing image cannot make the
        # surrounding layout repeatedly resize the canvas during a pinch.
        self.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored)
        self.source_pixmap=QPixmap();self.image_rect=QRect();self.pan_offset=QPointF();self.zoom_factor=1.0;self.zoom_target=1.0;self.pinch_start_zoom=1.0;self.grabGesture(Qt.PinchGesture)
        self.zoom_timer=QTimer(self);self.zoom_timer.setInterval(16);self.zoom_timer.timeout.connect(self._animate_zoom)
        self.long_press_timer=QTimer(self);self.long_press_timer.setSingleShot(True);self.long_press_timer.setInterval(550);self.long_press_timer.timeout.connect(self._show_segment_picker)
        self.press_origin=None;self.press_position=None;self.press_global=None;self.long_press_fired=False;self.brush_dragging=False;self.brush_radius=5;self.transform_bounds=None;self.transform_drag_mode=None;self.transform_last=None;self.transform_center=None
    def set_source_pixmap(self,pixmap):
        self.source_pixmap=pixmap;self._rescale()
    def clear(self):
        self.zoom_timer.stop();self.source_pixmap=QPixmap();self.image_rect=QRect();self.pan_offset=QPointF();self.zoom_factor=1.0;self.zoom_target=1.0;super().clear()
    def reset_zoom(self):
        self.zoom_timer.stop();self.zoom_factor=1.0;self.zoom_target=1.0;self.pan_offset=QPointF();self._rescale()
    def _rescale(self):
        if self.source_pixmap.isNull():return
        available=self.size()-QSize(32,32)
        if available.width()<=0 or available.height()<=0:return
        fitted=self.source_pixmap.scaled(available,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        target=QSize(max(1,round(fitted.width()*self.zoom_factor)),max(1,round(fitted.height()*self.zoom_factor)))
        scaled=self.source_pixmap.scaled(target,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self._clamp_pan(scaled.size())
        left=round((self.width()-scaled.width())/2+self.pan_offset.x());top=round((self.height()-scaled.height())/2+self.pan_offset.y())
        self.image_rect=QRect(left,top,scaled.width(),scaled.height())
        viewport=QPixmap(self.size());viewport.fill(Qt.transparent)
        painter=QPainter(viewport);painter.drawPixmap(self.image_rect.topLeft(),scaled);painter.end()
        if self.tool_mode=="transform" and self.transform_bounds is not None:
            bounds=self._transform_rect();rotation_handle=self._rotation_handle(bounds);painter=QPainter(viewport);painter.setRenderHint(QPainter.Antialiasing);painter.setPen(QPen(QColor("#48d17f"),2,Qt.DashLine));painter.setBrush(Qt.NoBrush);painter.drawRect(bounds);painter.setPen(QPen(QColor("#48d17f"),2));painter.drawLine(bounds.center().x(),bounds.top(),rotation_handle.center().x(),rotation_handle.center().y());painter.setBrush(QColor("#f5f7fb"));painter.drawRect(QRect(bounds.right()-6,bounds.bottom()-6,12,12));painter.drawEllipse(rotation_handle);painter.end()
        super().setPixmap(viewport)
        self._apply_tool_cursor()
    def resizeEvent(self,event):
        super().resizeEvent(event);self._rescale()
    def _zoom_by(self,factor):
        if factor<=0:return
        self._set_zoom_target(self.zoom_target*factor)
    def _set_zoom_target(self,zoom):
        zoom=min(8.0,max(1.0,zoom))
        # Suppress tiny direction reversals produced by noisy trackpad data.
        if abs(zoom-self.zoom_target)<.002:return
        self.zoom_target=zoom
        if not self.zoom_timer.isActive():self.zoom_timer.start()
    def _clamp_pan(self,image_size=None):
        if image_size is None:image_size=self.image_rect.size()
        max_x=max(0.0,(image_size.width()-self.width())/2);max_y=max(0.0,(image_size.height()-self.height())/2)
        self.pan_offset.setX(min(max(self.pan_offset.x(),-max_x),max_x));self.pan_offset.setY(min(max(self.pan_offset.y(),-max_y),max_y))
    def _pan_by(self,delta):
        if self.zoom_target<=1.0:return False
        before=QPointF(self.pan_offset);self.pan_offset+=QPointF(delta);self._clamp_pan()
        if self.pan_offset==before:return False
        self._rescale();return True
    def _animate_zoom(self):
        delta=self.zoom_target-self.zoom_factor
        if abs(delta)<.002:
            self.zoom_factor=self.zoom_target;self.zoom_timer.stop()
        else:self.zoom_factor+=delta*.35
        self._rescale()
    def event(self,event):
        if event.type()==QEvent.Gesture:
            pinch=event.gesture(Qt.PinchGesture)
            if pinch is not None:
                if pinch.state()==Qt.GestureStarted:self.pinch_start_zoom=self.zoom_target
                self._set_zoom_target(self.pinch_start_zoom*pinch.totalScaleFactor())
                event.accept();return True
        if event.type()==QEvent.NativeGesture and event.gestureType()==Qt.ZoomNativeGesture:
            # Exponential scaling treats equal pinch-in/out deltas symmetrically.
            self._zoom_by(float(np.exp(event.value())))
            event.accept();return True
        if event.type()==QEvent.NativeGesture and event.gestureType()==Qt.PanNativeGesture:
            if self._pan_by(event.delta()):event.accept();return True
        return super().event(event)
    def wheelEvent(self,event):
        # Trackpads normally deliver two-finger movement as pixel-precise wheel
        # events. Mouse wheels remain available to surrounding scroll areas.
        delta=event.pixelDelta()
        if not delta.isNull() and self._pan_by(delta):event.accept();return
        angle=event.angleDelta()
        if not angle.isNull() and self._pan_by(QPointF(angle.x()/8,angle.y()/8)):event.accept();return
        super().wheelEvent(event)
    def mouseDoubleClickEvent(self,event):
        if event.button()==Qt.LeftButton:
            self.reset_zoom();event.accept();return
        super().mouseDoubleClickEvent(event)
    def set_tool_mode(self,mode,inverse=False):
        self.tool_mode=mode;self.wand_inverse=inverse
        self._apply_tool_cursor();self._rescale()
    def set_brush_radius(self,radius):
        self.brush_radius=max(1,int(radius));self._apply_tool_cursor()
    def _apply_tool_cursor(self):
        if self.crop_mode:return
        if self.tool_mode in ("brush","pen"):
            source_width=self.source_pixmap.width() if not self.source_pixmap.isNull() else 0
            diameter=brush_cursor_diameter(self.brush_radius,self.image_rect.width(),source_width)
            extent=diameter+6;pm=QPixmap(extent,extent);pm.fill(Qt.transparent)
            painter=QPainter(pm);painter.setRenderHint(QPainter.Antialiasing);bounds=QRect(3,3,diameter,diameter)
            painter.setBrush(Qt.NoBrush);painter.setPen(QPen(QColor(0,0,0,190),3));painter.drawEllipse(bounds);painter.setPen(QPen(QColor("#ffffff"),1));painter.drawEllipse(bounds);painter.end()
            self.setCursor(QCursor(pm,extent//2,extent//2));return
        self.setCursor(Qt.PointingHandCursor if self.tool_mode=="wand" else (Qt.OpenHandCursor if self.tool_mode=="transform" else Qt.ArrowCursor))
    def set_transform_bounds(self,bounds):
        self.transform_bounds=bounds;self._rescale()
    def _transform_rect(self):
        if self.transform_bounds is None:return QRect()
        x,y,w,h=self.transform_bounds
        return QRect(round(self.image_rect.left()+x*self.image_rect.width()),round(self.image_rect.top()+y*self.image_rect.height()),max(1,round(w*self.image_rect.width())),max(1,round(h*self.image_rect.height())))
    def _rotation_handle(self,rect):
        return QRect(rect.center().x()-7,rect.top()-31,14,14)
    def _rotation_angle(self,point):
        return math.degrees(math.atan2(point.y()-self.transform_center.y(),point.x()-self.transform_center.x()))
    def set_crop_mode(self,enabled):
        self.crop_mode=enabled;self.crop_start=None;self.rubber_band.hide()
        self.setCursor(Qt.CrossCursor) if enabled else self._apply_tool_cursor()
    def mouseMoveEvent(self,e):
        if self.crop_mode and self.crop_start is not None:
            self.rubber_band.setGeometry(QRect(self.crop_start,self._crop_point(e)).normalized());return
        if self.transform_drag_mode and e.buttons()&Qt.LeftButton:
            if self.transform_drag_mode=="rotate":
                current=self._rotation_angle(e.position().toPoint());last=self.transform_last;delta=(current-last+180)%360-180;self.transform_last=current;self.transform_dragged.emit("rotate",delta,0);return
            current=self._normalized_point(e.position().toPoint());last=self.transform_last;self.transform_last=current
            if current and last:self.transform_dragged.emit(self.transform_drag_mode,current[0]-last[0],current[1]-last[1])
            return
        if self.press_position is not None and e.buttons()&Qt.LeftButton:
            if (e.position().toPoint()-self.press_origin).manhattanLength()>=QApplication.startDragDistance():
                self.long_press_timer.stop();self.press_position=None;self.brush_dragging=True;self.edit_started.emit();self._paint(e)
            return
        if supports_drag_paint(self.tool_mode) and e.buttons()&(Qt.LeftButton|Qt.RightButton):self._paint(e)
        else:self._hover(e)
    def mousePressEvent(self,e):
        if self.crop_mode:
            if e.button()==Qt.LeftButton:
                self.crop_start=self._crop_point(e);self.rubber_band.setGeometry(QRect(self.crop_start,QSize()));self.rubber_band.show()
            return
        position=self._position(e)
        if self.tool_mode=="transform" and e.button()==Qt.LeftButton and self.transform_bounds is not None:
            rect=self._transform_rect();point=e.position().toPoint();resize_handle=QRect(rect.right()-12,rect.bottom()-12,24,24);rotation_handle=self._rotation_handle(rect)
            if rotation_handle.adjusted(-5,-5,5,5).contains(point):
                self.transform_drag_mode="rotate";self.transform_center=rect.center();self.transform_last=self._rotation_angle(point);self.edit_started.emit();self.setCursor(Qt.CrossCursor);e.accept();return
            if position and (resize_handle.contains(point) or rect.contains(point)):
                self.transform_drag_mode="resize" if resize_handle.contains(point) else "move";self.transform_last=position;self.edit_started.emit();self.setCursor(Qt.SizeFDiagCursor if self.transform_drag_mode=="resize" else Qt.ClosedHandCursor);e.accept();return
        if position and self.tool_mode=="wand":
            inverse=self.wand_inverse or e.button()==Qt.RightButton
            self.wand_requested.emit(int(position[0]*10000),int(position[1]*10000),inverse);return
        modifiers=e.modifiers()
        if position and modifiers&Qt.ShiftModifier:
            self.segment_requested.emit(int(position[0]*10000),int(position[1]*10000),"remove");return
        if position and modifiers&(Qt.ControlModifier|Qt.MetaModifier):
            # Qt maps Command to ControlModifier and Control to MetaModifier on macOS.
            mode="add" if (sys.platform=="darwin" and modifiers&Qt.MetaModifier) or (sys.platform!="darwin" and modifiers&Qt.ControlModifier) else "replace"
            self.segment_requested.emit(int(position[0]*10000),int(position[1]*10000),mode);return
        if position and e.button()==Qt.LeftButton:
            self.press_origin=e.position().toPoint();self.press_position=position;self.press_global=e.globalPosition().toPoint();self.long_press_fired=False;self.brush_dragging=False;self.long_press_timer.start();e.accept();return
        if position:self.edit_started.emit()
        self._paint(e)
    def mouseReleaseEvent(self,e):
        if self.crop_mode and self.crop_start is not None:
            end=self._crop_point(e);start=self.crop_start;self.set_crop_mode(False)
            a=self._normalized_point(start);b=self._normalized_point(end)
            self.crop_requested.emit(int(a[0]*10000),int(a[1]*10000),int(b[0]*10000),int(b[1]*10000));return
        if e.button()==Qt.LeftButton and self.transform_drag_mode:
            self.transform_drag_mode=None;self.transform_last=None;self.transform_center=None;self.setCursor(Qt.OpenHandCursor);e.accept();return
        if e.button()==Qt.LeftButton:
            self.long_press_timer.stop()
            if self.long_press_fired:
                self.long_press_fired=False;e.accept();return
            if self.press_position is not None:
                position=self._position(e);self.press_position=None
                if position:self.edit_started.emit();self.painted.emit(int(position[0]*10000),int(position[1]*10000),False)
                e.accept();return
            if self.brush_dragging:self.brush_dragging=False;e.accept();return
        super().mouseReleaseEvent(e)
    def leaveEvent(self,e):self.hover_left.emit();super().leaveEvent(e)
    def _position(self,e):
        return self._normalized_point(e.position().toPoint(),False)
    def _normalized_point(self,point,clamp=True):
        if self.source_pixmap.isNull() or self.image_rect.isEmpty():return None
        x=(point.x()-self.image_rect.left())/self.image_rect.width();y=(point.y()-self.image_rect.top())/self.image_rect.height()
        if clamp:return (min(max(x,0),1),min(max(y,0),1))
        return (x,y) if 0<=x<=1 and 0<=y<=1 else None
    def _crop_point(self,e):
        point=e.position().toPoint()
        if self.source_pixmap.isNull() or self.image_rect.isEmpty():return point
        point.setX(min(max(point.x(),self.image_rect.left()),self.image_rect.left()+self.image_rect.width()));point.setY(min(max(point.y(),self.image_rect.top()),self.image_rect.top()+self.image_rect.height()))
        return point
    def _hover(self,e):
        position=self._position(e)
        if position:self.hovered.emit(int(position[0]*10000),int(position[1]*10000))
        else:self.hover_left.emit()
    def _paint(self,e):
        position=self._position(e)
        if position:self.painted.emit(int(position[0]*10000),int(position[1]*10000),bool(e.buttons()&Qt.RightButton))
    def _show_segment_picker(self):
        if self.press_position is None:return
        position=self.press_position;global_position=self.press_global
        self.press_position=None;self.long_press_fired=True
        self.segment_picker_requested.emit(int(position[0]*10000),int(position[1]*10000),global_position)

class SegmentList(QListWidget):
    """Multi-select list where Shift-click removes instead of selecting a range."""
    remove_requested=Signal(int)
    row_hovered=Signal(int);hover_left=Signal()
    def __init__(self):
        super().__init__();self.setMouseTracking(True)
    def mouseMoveEvent(self,event):
        item=self.itemAt(event.position().toPoint())
        self.row_hovered.emit(self.row(item) if item else -1)
        super().mouseMoveEvent(event)
    def leaveEvent(self,event):
        self.hover_left.emit();super().leaveEvent(event)
    def mousePressEvent(self,event):
        if event.modifiers()&Qt.ShiftModifier:
            item=self.itemAt(event.position().toPoint())
            if item:
                item.setSelected(False);self.setCurrentItem(item,QItemSelectionModel.NoUpdate);event.accept();return
        control=(event.modifiers()&Qt.MetaModifier) if sys.platform=="darwin" else (event.modifiers()&Qt.ControlModifier)
        if control:
            item=self.itemAt(event.position().toPoint())
            if item:
                item.setSelected(True);self.setCurrentItem(item,QItemSelectionModel.NoUpdate);event.accept();return
        super().mousePressEvent(event)
    def contextMenuEvent(self,event):
        item=self.itemAt(event.pos())
        if not item:return
        row=self.row(item);self.setCurrentItem(item,QItemSelectionModel.ClearAndSelect)
        menu=QMenu(self);remove=menu.addAction("Remove segment")
        if menu.exec(event.globalPos())==remove:self.remove_requested.emit(row)

class AIWorker(QObject):
    progress=Signal(str); finished=Signal(object,object); failed=Signal(str)
    def __init__(self,image,layer_count,points=None,quality="accurate",minimum_segment_area=0): super().__init__(); self.image=image; self.layer_count=layer_count; self.points=points; self.quality=quality; self.minimum_segment_area=minimum_segment_area
    @Slot()
    def run(self):
        try:
            from .ai import LocalAI
            depth,segments=LocalAI(segmentation_quality=self.quality,minimum_segment_area=self.minimum_segment_area).segment_image(self.image,self.progress.emit,self.points); self.finished.emit(depth,segments)
        except Exception as exc: self.failed.emit(str(exc))

class KeybindingsDialog(QDialog):
    def __init__(self,bindings,parent=None):
        super().__init__(parent);self.setWindowTitle("Keybindings");self.setMinimumWidth(520);self.editors={}
        layout=QVBoxLayout(self);intro=QLabel("Assign a key or key combination to any command. Unassigned commands have no shortcut.");intro.setWordWrap(True);layout.addWidget(intro)
        form=QFormLayout();form.setVerticalSpacing(9)
        for command_id,label in COMMANDS:
            row=QWidget();row_layout=QHBoxLayout(row);row_layout.setContentsMargins(0,0,0,0);row_layout.setSpacing(6)
            editor=QKeySequenceEdit(QKeySequence(bindings.get(command_id,"")));editor.setClearButtonEnabled(True);self.editors[command_id]=editor
            clear=QPushButton("Clear",objectName="ghost");clear.setFixedWidth(62);clear.clicked.connect(editor.clear)
            row_layout.addWidget(editor,1);row_layout.addWidget(clear);form.addRow(label,row)
        layout.addLayout(form)
        clear_all=QPushButton("Clear all",objectName="ghost");clear_all.clicked.connect(lambda:[editor.clear() for editor in self.editors.values()]);layout.addWidget(clear_all,0,Qt.AlignLeft)
        buttons=QDialogButtonBox(QDialogButtonBox.Cancel|QDialogButtonBox.Save);buttons.accepted.connect(self._accept_bindings);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
    def _accept_bindings(self):
        bindings=self.bindings();used={}
        for command_id,shortcut in bindings.items():
            if shortcut in used:
                labels=dict(COMMANDS);QMessageBox.warning(self,"Duplicate keybinding",f"{shortcut} is assigned to both {labels[used[shortcut]]} and {labels[command_id]}.");return
            used[shortcut]=command_id
        self.accept()
    def bindings(self):
        return {command_id:text for command_id,editor in self.editors.items() if (text:=editor.keySequence().toString(QKeySequence.PortableText))}

class ExportDimensionsDialog(QDialog):
    """Collect the shared physical size applied to every exported STL."""
    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle("STL dimensions")
        layout=QVBoxLayout(self);layout.addWidget(QLabel("Enter the dimensions for each STL file."))
        form=QFormLayout();width,height,depth=DEFAULT_STL_DIMENSIONS_MM;self.width=self._dimension(width);self.height=self._dimension(height);self.depth=self._dimension(depth)
        form.addRow("Width",self.width);form.addRow("Height",self.height);form.addRow("Depth",self.depth);layout.addLayout(form)
        buttons=QDialogButtonBox(QDialogButtonBox.Cancel|QDialogButtonBox.Ok);buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
    def _dimension(self,value):
        box=QDoubleSpinBox();box.setRange(.01,10000);box.setDecimals(2);box.setValue(value);box.setSuffix(" mm");return box
    def dimensions(self):
        return self.width.value(),self.height.value(),self.depth.value()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.project=None; self.project_path=None; self.view="Image";self.pre_transform_view="Image"; self.connector_cache=ConnectorCache(); self.hovered_segment=-1;self.undo_stack=[];self.redo_stack=[];self.history_limit=30; self.setWindowTitle("LumaLayer — Hybrid Lightbox Designer")
        session_dir=Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation));self.session_path=session_dir/"last-session.hbox";self.keybinding_store=KeybindingStore(session_dir/"keybindings.json");self.keybindings={**DEFAULT_BINDINGS,**self.keybinding_store.load()};self.keybinding_shortcuts=[]
        self.autosave_timer=QTimer(self);self.autosave_timer.setSingleShot(True);self.autosave_timer.setInterval(500);self.autosave_timer.timeout.connect(self._save_session)
        self.resize(1280,800); self.setMinimumSize(980,650); self.setStyleSheet(STYLE); self._build_ui(); self._build_menu();self._apply_keybindings(); self._set_enabled(False);self._restore_session()
    def _build_ui(self):
        root=QWidget(objectName="root"); outer=QVBoxLayout(root); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)
        top=QFrame(objectName="topbar"); top.setFixedHeight(64); tl=QHBoxLayout(top); tl.setContentsMargins(22,10,22,10)
        logo=QLabel("◈"); logo.setStyleSheet("font-size:26px;color:#8b6cff;"); titles=QVBoxLayout(); titles.setSpacing(0)
        titles.addWidget(QLabel("LumaLayer",objectName="brand")); titles.addWidget(QLabel("HYBRID LIGHTBOX DESIGNER",objectName="eyebrow")); tl.addWidget(logo); tl.addLayout(titles); tl.addStretch()
        self.new_btn=QPushButton("New"); self.new_btn.clicked.connect(self.new_project);self.open_btn=QPushButton("Load"); self.open_btn.clicked.connect(self.open_project); self.save_btn=QPushButton("Save"); self.save_btn.clicked.connect(self.save_project);self.undo_btn=QPushButton("↶ Undo",objectName="ghost");self.undo_btn.clicked.connect(self.undo);self.redo_btn=QPushButton("↷ Redo",objectName="ghost");self.redo_btn.clicked.connect(self.redo)
        self.export_btn=QPushButton("Export STLs",objectName="primary"); self.export_btn.clicked.connect(self.export_stls); tl.addWidget(self.new_btn);tl.addWidget(self.open_btn); tl.addWidget(self.save_btn);tl.addSpacing(8);tl.addWidget(self.undo_btn);tl.addWidget(self.redo_btn); tl.addWidget(self.export_btn); outer.addWidget(top)
        body=QHBoxLayout(); body.setContentsMargins(16,16,16,14); body.setSpacing(14)
        left=QFrame(objectName="sidebar"); left.setFixedWidth(260);left_shell=QVBoxLayout(left);left_shell.setContentsMargins(1,1,1,1);left_shell.setSpacing(0);sidebar_scroll=QScrollArea(objectName="sidebarScroll");sidebar_scroll.setWidgetResizable(True);sidebar_scroll.setFrameShape(QFrame.NoFrame);sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);sidebar_content=QWidget();ll=QVBoxLayout(sidebar_content);ll.setContentsMargins(13,15,13,13);ll.setSpacing(10);ll.setSizeConstraint(QLayout.SetMinAndMaxSize);sidebar_scroll.setWidget(sidebar_content);left_shell.addWidget(sidebar_scroll);self.segments_toggle=QPushButton("▸  SEGMENTS",objectName="ghost");self.segments_toggle.setCheckable(True);self.segments_toggle.setChecked(False);self.segments_toggle.setStyleSheet("text-align:left;");self.segments_toggle.toggled.connect(self.toggle_segments);ll.addWidget(self.segments_toggle);self.segments_content=QWidget();segments_layout=QVBoxLayout(self.segments_content);segments_layout.setContentsMargins(0,0,0,0);segments_layout.setSpacing(10)
        ll.addWidget(QLabel("SOURCE IMAGES",objectName="heading"));self.source_images=QListWidget();self.source_images.setMaximumHeight(132);self.source_images.currentRowChanged.connect(self.select_source_image);ll.addWidget(self.source_images);source_actions=QHBoxLayout();self.add_source_button=QPushButton("＋ Add image");self.add_source_button.clicked.connect(self.add_source_image);self.remove_source_button=QPushButton("− Remove");self.remove_source_button.clicked.connect(self.remove_source_image);source_actions.addWidget(self.add_source_button);source_actions.addWidget(self.remove_source_button);ll.addLayout(source_actions)
        self.segments=SegmentList(); self.segments.setSelectionMode(QAbstractItemView.ExtendedSelection); self.segments.setIconSize(QSize(18,18)); self.segments.currentRowChanged.connect(self.select_segment); self.segments.itemSelectionChanged.connect(self.segment_selection_changed); self.segments.remove_requested.connect(self.remove_segment);self.segments.row_hovered.connect(self.hover_segment_index);self.segments.hover_left.connect(self.clear_hover); segments_layout.addWidget(self.segments,1)
        segment_actions=QHBoxLayout(); self.replace_segment=QPushButton("Replace"); self.add_segment=QPushButton("Add"); self.subtract_segment=QPushButton("Erase"); self.replace_segment.clicked.connect(lambda:self.apply_segment("replace")); self.add_segment.clicked.connect(lambda:self.apply_segment("add")); self.subtract_segment.clicked.connect(lambda:self.apply_segment("erase")); segment_actions.addWidget(self.replace_segment); segment_actions.addWidget(self.add_segment); segment_actions.addWidget(self.subtract_segment); segments_layout.addLayout(segment_actions);self.segments_content.hide();ll.addWidget(self.segments_content,1)
        width_filter=QHBoxLayout();width_filter.addWidget(QLabel("Min print width",objectName="muted"));self.minimum_print_width=self.spin(0,10,.4," mm");self.minimum_print_width.setToolTip("Remove regions narrower than this physical width from every layer and segment mask");width_filter.addWidget(self.minimum_print_width,1);ll.addLayout(width_filter);self.remove_thin_button=QPushButton("Clean all masks");self.remove_thin_button.setToolTip("Apply the minimum print width to every physical layer and generated segment mask");self.remove_thin_button.clicked.connect(self.remove_thin_all_masks);ll.addWidget(self.remove_thin_button)
        self.ai_quality=QComboBox(); self.ai_quality.addItems(["Accurate (Large)","Base (Base+)","Fast (Tiny)"]); ll.addWidget(self.ai_quality)
        segment_filter=QHBoxLayout(); segment_filter.addWidget(QLabel("Min segment",objectName="muted")); self.minimum_segment_area=QSpinBox(); self.minimum_segment_area.setRange(0,10_000_000); self.minimum_segment_area.setValue(100); self.minimum_segment_area.setSuffix(" px"); self.minimum_segment_area.setSpecialValueText("Disabled"); self.minimum_segment_area.setToolTip("Discard generated segments smaller than this pixel area; set to 0 to disable"); segment_filter.addWidget(self.minimum_segment_area,1); ll.addLayout(segment_filter)
        self.ai_button=QPushButton("◉  Segment image",objectName="primary"); self.ai_button.clicked.connect(self.run_ai); ll.addWidget(self.ai_button)
        self.auto_layer_button=QPushButton("✦  Auto-assign to layers"); self.auto_layer_button.clicked.connect(self.auto_assign_layers); ll.addWidget(self.auto_layer_button)
        self.optimize=QPushButton("✦  Optimize light paths"); self.optimize.setToolTip(f"Uses a fixed {OPTIMIZER_NOZZLE_DIAMETER_MM:g} mm nozzle diameter"); self.optimize.clicked.connect(self.run_optimize); self.add_connectors_button=QPushButton("＋  Add connectors");self.add_connectors_button.setToolTip("Generate and visualize the structural connectors that will be included during export");self.add_connectors_button.clicked.connect(self.add_connectors); self.validate=QPushButton("✓  Validate structure",objectName="ghost"); self.validate.clicked.connect(self.run_validate); ll.addWidget(self.optimize);ll.addWidget(self.add_connectors_button); ll.addWidget(self.validate)
        stage=QFrame(objectName="stage"); sl=QVBoxLayout(stage); sl.setContentsMargins(12,12,12,12); sl.setSpacing(10)
        main_view=QWidget();main_view_layout=QVBoxLayout(main_view);main_view_layout.setContentsMargins(0,0,0,0);main_view_layout.setSpacing(5);sh=QHBoxLayout();self.stage_title=QLabel("Canvas",objectName="heading");self.stage_meta=QLabel("No image loaded",objectName="muted");sh.addWidget(self.stage_title);sh.addStretch();sh.addWidget(self.stage_meta);main_view_layout.addLayout(sh)
        self.canvas_stack=QStackedWidget(); self.canvas=Canvas(); self.canvas.edit_started.connect(self.begin_brush_history);self.canvas.painted.connect(self.paint_mask); self.canvas.wand_requested.connect(self.apply_wand);self.canvas.transform_dragged.connect(self.mouse_transform_layer); self.canvas.hovered.connect(self.hover_segment); self.canvas.hover_left.connect(self.clear_hover); self.canvas.segment_requested.connect(self.select_segment_at);self.canvas.segment_picker_requested.connect(self.show_segment_picker); self.canvas.crop_requested.connect(self.apply_crop); empty=QWidget(); el=QVBoxLayout(empty); el.setAlignment(Qt.AlignCenter)
        icon=QLabel("◫"); icon.setAlignment(Qt.AlignCenter); icon.setStyleSheet("font-size:54px;color:#6551c8;"); title=QLabel("Start with an image",objectName="emptyTitle"); title.setAlignment(Qt.AlignCenter); copy=QLabel("Import a JPEG, PNG, or WebP to create three editable layers.",objectName="emptyCopy"); copy.setAlignment(Qt.AlignCenter)
        choose=QPushButton("Choose image…",objectName="primary"); choose.clicked.connect(self.open_image); choose.setFixedWidth(150); el.addStretch(); el.addWidget(icon); el.addWidget(title); el.addWidget(copy); el.addSpacing(8); el.addWidget(choose,0,Qt.AlignCenter); el.addStretch(); self.canvas_stack.addWidget(empty); self.canvas_stack.addWidget(self.canvas);main_view_layout.addWidget(self.canvas_stack,1)
        illustration=QWidget();illustration.setMinimumWidth(220);illustration_layout=QVBoxLayout(illustration);illustration_layout.setContentsMargins(0,0,0,0);illustration_layout.setSpacing(5);illustration_layout.addWidget(QLabel("ILLUSTRATION",objectName="heading"));self.illustration=PreviewLabel();self.illustration.setMinimumSize(210,135);self.illustration.setStyleSheet("background:#0c0e12;border:1px solid #303642;border-radius:8px;");illustration_layout.addWidget(self.illustration,1)
        self.preview_splitter=QSplitter(Qt.Horizontal);self.preview_splitter.setObjectName("previewSplitter");self.preview_splitter.setChildrenCollapsible(False);self.preview_splitter.setHandleWidth(8);self.preview_splitter.addWidget(main_view);self.preview_splitter.addWidget(illustration);self.preview_splitter.setStretchFactor(0,3);self.preview_splitter.setStretchFactor(1,1);self.preview_splitter.setSizes([720,240]);sl.addWidget(self.preview_splitter,1)
        tools=QFrame(objectName="toolstrip"); tr=QHBoxLayout(tools); tr.setContentsMargins(8,6,8,6); self.view_box=QComboBox(); self.view_box.addItems(["Image","Segmentations","Selected segment","Layer segments","Connector view","Depth map","Selected mask","Backlit preview"]); self.view_box.currentTextChanged.connect(self.set_view); self.tool_box=QComboBox();self.tool_box.addItems(["Brush","Magic wand","Transform","Pen"]);self.tool_box.currentIndexChanged.connect(self.tool_changed);self.brush=QSlider(Qt.Horizontal); self.brush.setRange(1,30); self.brush.setValue(5); self.brush.setFixedWidth(120); self.brush.setToolTip("Brush radius in pixels"); self.brush_value=QLabel("5 px",objectName="muted"); self.brush_value.setMinimumWidth(48); self.brush.valueChanged.connect(self.tool_value_changed);self.wand_inverse=QCheckBox("Inverse (deselect)");self.wand_inverse.toggled.connect(self.inverse_changed);self.wand_inverse.hide()
        self.crop_button=QPushButton("Crop");self.crop_button.clicked.connect(self.begin_crop);self.subtract_button=QPushButton("Subtract");self.subtract_button.setToolTip("Subtract another layer's mask from the current layer");self.subtract_button.clicked.connect(self.subtract_layer);self.add_layer_button=QPushButton("Add layer…");self.add_layer_button.setToolTip("Choose a layer whose mask will be added to the current layer");self.add_layer_button.clicked.connect(self.add_selected_layer);self.fill_gaps_button=QPushButton("Fill gaps");self.fill_gaps_button.setToolTip("Cover every image pixel not already covered by previous layers, with an optional overlap margin");self.fill_gaps_button.clicked.connect(self.fill_gaps)
        self.tool_hint=QLabel("Drag to paint  ·  Right-drag to erase",objectName="muted");tr.addWidget(QLabel("View",objectName="muted")); tr.addWidget(self.view_box);tr.addWidget(self.crop_button); tr.addSpacing(10);tr.addWidget(self.tool_box); tr.addWidget(self.brush); tr.addWidget(self.brush_value);tr.addWidget(self.subtract_button);tr.addWidget(self.add_layer_button);tr.addWidget(self.fill_gaps_button);tr.addWidget(self.wand_inverse); tr.addStretch(); tr.addWidget(self.tool_hint); sl.addWidget(tools)
        layer_dock=QFrame(objectName="layerDock");layer_dock_layout=QHBoxLayout(layer_dock);layer_dock_layout.setContentsMargins(10,8,10,8);layer_dock_layout.setSpacing(12);layer_browser=QWidget();layer_browser_layout=QVBoxLayout(layer_browser);layer_browser_layout.setContentsMargins(0,0,0,0);layer_browser_layout.setSpacing(5);layer_header=QHBoxLayout();self.layer_heading=QLabel("LAYERS",objectName="heading");layer_header.addWidget(self.layer_heading);layer_header.addStretch();self.full_mask_button=QPushButton("Full mask",objectName="ghost");self.full_mask_button.setToolTip("Select the entire image in the current layer");self.full_mask_button.clicked.connect(self.full_mask);layer_header.addWidget(self.full_mask_button);self.clear_layer_button=QPushButton("Clear selected layer",objectName="ghost");self.clear_layer_button.clicked.connect(self.clear_layer);layer_header.addWidget(self.clear_layer_button);layer_browser_layout.addLayout(layer_header)
        self.layers=LayerList(objectName="layerStrip");self.layers.setItemDelegate(LayerItemDelegate(self.layers));self.layers.setViewMode(QListView.IconMode);self.layers.setFlow(QListView.LeftToRight);self.layers.setWrapping(False);self.layers.setResizeMode(QListView.Adjust);self.layers.setMovement(QListView.Static);self.layers.setDragDropMode(QAbstractItemView.InternalMove);self.layers.setDragEnabled(True);self.layers.setAcceptDrops(True);self.layers.viewport().setAcceptDrops(True);self.layers.setDefaultDropAction(Qt.MoveAction);self.layers.setDropIndicatorShown(False);self.layers.setGridSize(QSize(138,146));self.layers.setIconSize(QSize(112,102));self.layers.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel);self.layers.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff);self.layers.currentRowChanged.connect(self.refresh_controls);self.layers.add_requested.connect(self.add_layer);self.layers.remove_requested.connect(self.remove_layer);self.layers.move_requested.connect(self.move_layer);self.layers.link_toggled.connect(self.toggle_layer_link);layer_browser_layout.addWidget(self.layers,1);layer_dock_layout.addWidget(layer_browser,2)
        layer_dock.setFixedHeight(205);sl.addWidget(layer_dock)
        inspector=QWidget();il=QVBoxLayout(inspector);il.setContentsMargins(0,0,0,0);il.setSpacing(10);self.layer_settings_toggle=QPushButton("▸  LAYER SETTINGS",objectName="ghost");self.layer_settings_toggle.setCheckable(True);self.layer_settings_toggle.setChecked(False);self.layer_settings_toggle.setStyleSheet("text-align:left;");self.layer_settings_toggle.toggled.connect(self.toggle_layer_settings);il.addWidget(self.layer_settings_toggle);self.layer_settings_content=QWidget();layer_settings_layout=QVBoxLayout(self.layer_settings_content);layer_settings_layout.setContentsMargins(0,0,0,0);layer_settings_layout.setSpacing(12);self.selected_name=QLabel("Selected layer",objectName="heading");layer_settings_layout.addWidget(self.selected_name);self.apply_settings_to_all=QCheckBox("Apply settings to all layers");self.apply_settings_to_all.setChecked(True);self.apply_settings_to_all.setToolTip("When unchecked, each layer keeps its own behavior and material settings");self.apply_settings_to_all.toggled.connect(self.apply_all_layers_toggled);layer_settings_layout.addWidget(self.apply_settings_to_all)
        form=QFormLayout(); form.setVerticalSpacing(11); self.mode=QComboBox(); self.mode.addItems([x.value.title() for x in LayerMode]); self.mode.currentTextChanged.connect(self.settings_changed); form.addRow("Behavior",self.mode)
        self.min_t=self.spin(.2,10,.4," mm"); self.max_t=self.spin(.2,10,1.6," mm"); self.mu=self.spin(.01,5,.75); self.gamma=self.spin(.1,5,1); form.addRow("Minimum",self.min_t); form.addRow("Maximum",self.max_t); form.addRow("Attenuation",self.mu); form.addRow("Gamma",self.gamma); layer_settings_layout.addLayout(form);self.layer_settings_content.hide();il.addWidget(self.layer_settings_content); il.addSpacing(8);self.project_settings_toggle=QPushButton("▸  PROJECT",objectName="ghost");self.project_settings_toggle.setCheckable(True);self.project_settings_toggle.setChecked(False);self.project_settings_toggle.setStyleSheet("text-align:left;");self.project_settings_toggle.toggled.connect(self.toggle_project_settings);il.addWidget(self.project_settings_toggle);self.project_settings_content=QWidget();project_settings_layout=QVBoxLayout(self.project_settings_content);project_settings_layout.setContentsMargins(0,0,0,0); pf=QFormLayout(); self.width_mm=self.spin(1,500,120," mm"); self.mesh_res=self.spin(.05,1,.12," mm"); self.mesh_res.setToolTip("Fine geometry can be sampled below nozzle width to preserve curves and narrow gaps"); self.print_smoothing=self.spin(0,3,0," mm"); self.print_smoothing.setToolTip("Rounds noisy mask edges and smooths thickness changes during STL export; leave at 0 for maximum detail"); self.structure=QComboBox(); self.structure.addItems(["Hidden bridges","Solid backing","Legacy bridges"]); self.structure.currentIndexChanged.connect(self.settings_changed); self.backing=self.spin(.2,1.5,.6," mm"); self.frame_width=self.spin(.5,10,1," mm"); self.frame_thickness=self.spin(.8,10,3," mm"); pf.addRow("Print width",self.width_mm); pf.addRow("Mesh detail",self.mesh_res); pf.addRow("Print smoothing",self.print_smoothing); pf.addRow("Structure",self.structure); pf.addRow("Backing",self.backing); pf.addRow("Border width",self.frame_width); pf.addRow("Border thickness",self.frame_thickness);project_settings_layout.addLayout(pf);self.project_settings_content.hide();il.addWidget(self.project_settings_content); il.addStretch()
        transform_form=QFormLayout();self.transform_x=QDoubleSpinBox();self.transform_y=QDoubleSpinBox();self.transform_scale_x=QDoubleSpinBox();self.transform_scale_y=QDoubleSpinBox();self.transform_rotation=QDoubleSpinBox()
        for widget in (self.transform_x,self.transform_y):widget.setRange(-10000,10000);widget.setSuffix(" px");widget.valueChanged.connect(self.transform_changed)
        for widget in (self.transform_scale_x,self.transform_scale_y):widget.setRange(.05,10);widget.setSingleStep(.05);widget.setValue(1);widget.setSuffix("×");widget.valueChanged.connect(self.transform_changed)
        self.transform_rotation.setRange(-180,180);self.transform_rotation.setSingleStep(1);self.transform_rotation.setSuffix("°");self.transform_rotation.valueChanged.connect(self.transform_changed)
        transform_form.addRow("Output X",self.transform_x);transform_form.addRow("Output Y",self.transform_y);transform_form.addRow("Output width",self.transform_scale_x);transform_form.addRow("Output height",self.transform_scale_y);transform_form.addRow("Rotation",self.transform_rotation);layer_settings_layout.addLayout(transform_form)
        note=QLabel("TIP\nOverlapping masks are intentional. Optimize Light Paths removes unnecessary material from rear layers.",objectName="muted"); note.setWordWrap(True); note.setStyleSheet("background:#1e222b;border:1px solid #2d333f;border-radius:9px;padding:11px;color:#8e98aa;"); il.addWidget(note)
        ll.addWidget(inspector);body.addWidget(left);body.addWidget(stage,1);outer.addLayout(body,1);self.setCentralWidget(root);self.setStatusBar(QStatusBar());self.statusBar().showMessage("Ready — open an image to begin")
    def _build_menu(self):
        menu=self.menuBar().addMenu("File")
        for label,key,slot in [("New project…","Ctrl+N",self.new_project),("Open image…","Ctrl+O",self.open_image),("Load project…","Ctrl+Shift+O",self.open_project),("Save project","Ctrl+S",self.save_project),("Save project as…","Ctrl+Shift+S",self.save_project_as),("Reset workspace",None,self.reset_workspace),("Export STLs…","Ctrl+E",self.export_stls)]:
            action=QAction(label,self)
            if key:action.setShortcut(key)
            action.triggered.connect(slot); menu.addAction(action)
        edit=self.menuBar().addMenu("Edit")
        self.history_actions={}
        for command_id,label,slot in [("history.undo","Undo",self.undo),("history.redo","Redo",self.redo)]:
            action=QAction(label,self);action.triggered.connect(slot);edit.addAction(action);self.history_actions[command_id]=action
        edit.addSeparator();keybindings=QAction("Keybindings…",self);keybindings.triggered.connect(self.show_keybindings);edit.addAction(keybindings)
    def show_keybindings(self):
        dialog=KeybindingsDialog(self.keybindings,self)
        if dialog.exec()!=QDialog.Accepted:return
        self.keybindings=dialog.bindings();self.keybinding_store.save(self.keybindings);self._apply_keybindings();self.statusBar().showMessage("Keybindings saved")
    def _apply_keybindings(self):
        for shortcut in self.keybinding_shortcuts:shortcut.setParent(None);shortcut.deleteLater()
        self.keybinding_shortcuts=[]
        for command_id,action in self.history_actions.items():action.setShortcut(self.keybindings.get(command_id,DEFAULT_BINDINGS[command_id]))
        for command_id,key in self.keybindings.items():
            if command_id in self.history_actions:continue
            shortcut=QShortcut(QKeySequence(key),self);shortcut.setContext(Qt.WindowShortcut);shortcut.setAutoRepeat(False);shortcut.activated.connect(lambda command_id=command_id:self.execute_keybinding(command_id));self.keybinding_shortcuts.append(shortcut)
    def execute_keybinding(self,command_id):
        dispatch_keybinding(command_id,self.tool_box.setCurrentText,self.view_box.setCurrentText,self.apply_segment,self.undo,self.redo)
    def spin(self,lo,hi,value,suffix=""):
        box=QDoubleSpinBox(); box.setRange(lo,hi); box.setDecimals(2); box.setValue(value); box.setSuffix(suffix); box.valueChanged.connect(self.settings_changed); return box
    def toggle_layer_settings(self,expanded):
        self.layer_settings_content.setVisible(expanded)
        self.layer_settings_toggle.setText(("▾" if expanded else "▸")+"  LAYER SETTINGS")
    def apply_all_layers_toggled(self,checked):
        if not self.project:return
        self.selected_name.setText("All layers" if checked else self.project.layers[self.current_layer_index()].name)
        if checked:self.settings_changed()
    def toggle_project_settings(self,expanded):
        self.project_settings_content.setVisible(expanded)
        self.project_settings_toggle.setText(("▾" if expanded else "▸")+"  PROJECT")
    def toggle_segments(self,expanded):
        self.segments_content.setVisible(expanded)
        self.segments_toggle.setText(("▾" if expanded else "▸")+"  SEGMENTS")
    def _set_enabled(self,enabled):
        for w in (self.save_btn,self.export_btn,self.layers,self.source_images,self.add_source_button,self.remove_source_button,self.segments,self.full_mask_button,self.clear_layer_button,self.ai_quality,self.minimum_segment_area,self.minimum_print_width,self.ai_button,self.auto_layer_button,self.optimize,self.add_connectors_button,self.validate,self.view_box,self.crop_button,self.tool_box,self.brush,self.subtract_button,self.add_layer_button,self.fill_gaps_button,self.wand_inverse,self.mode,self.min_t,self.max_t,self.mu,self.gamma,self.apply_settings_to_all,self.transform_x,self.transform_y,self.transform_scale_x,self.transform_scale_y,self.transform_rotation,self.width_mm,self.mesh_res,self.print_smoothing,self.structure,self.backing,self.frame_width,self.frame_thickness): w.setEnabled(enabled)
        if enabled and self.project:self.remove_source_button.setEnabled(len(self.project.sources)>1)
        self._update_segment_actions()
        self._update_history_actions()
    def open_image(self):
        path,_=QFileDialog.getOpenFileName(self,"Open image","","Images (*.jpg *.jpeg *.png *.webp)")
        if not path:return
        choices=["Standard — 768 px","High — 1536 px","Ultra — 3072 px","Original resolution"]
        choice,accepted=QInputDialog.getItem(self,"Import quality","Choose the maximum image dimension. Higher quality uses more memory and makes AI processing slower.",choices,1,False)
        if not accepted:return
        size={choices[0]:768,choices[1]:1536,choices[2]:3072,choices[3]:None}[choice]
        self.project_path=None;self.set_project(Project.from_image(path,size=size)); self.statusBar().showMessage(f"Opened {Path(path).name} at {self.project.image.shape[1]} × {self.project.image.shape[0]} px");self._schedule_session_save()
    def new_project(self):
        self.open_image()
    def add_source_image(self):
        if not self.project:return
        paths,_=QFileDialog.getOpenFileNames(self,"Add source images","","Images (*.jpg *.jpeg *.png *.webp)")
        if not paths:return
        self.record_history()
        for path in paths:self.project.add_source(path)
        self._populate_source_images();self.source_images.setCurrentRow(len(self.project.sources)-1);self.statusBar().showMessage(f"Added {len(paths)} source image(s)");self._schedule_session_save()
    def remove_source_image(self):
        if not self.project or len(self.project.sources)<=1:return
        index=self.source_images.currentRow()
        if not 0<=index<len(self.project.sources):return
        source=self.project.sources[index];affected=[layer.name for layer in self.project.layers if layer.source_index==index]
        detail=(f"\n\nThis will clear the masks for: {', '.join(affected)}." if affected else "")
        answer=QMessageBox.question(self,"Remove source image",f"Remove {source.name}?{detail}",QMessageBox.Yes|QMessageBox.Cancel,QMessageBox.Cancel)
        if answer!=QMessageBox.Yes:return
        self.record_history();cleared=self.project.remove_source(index);self.hovered_segment=-1;self.connector_cache.invalidate();self._populate_source_images();self._populate_segments();self._update_layer_previews();self.refresh();self._schedule_session_save()
        self.statusBar().showMessage(f"Removed {source.name}"+(f" and cleared {len(cleared)} layer mask(s)" if cleared else ""))
    def _populate_source_images(self):
        self.source_images.blockSignals(True);self.source_images.clear()
        if self.project:
            for i,source in enumerate(self.project.sources):self.source_images.addItem(f"{i+1}. {source.name}  ·  {len(source.segment_masks)} segments")
            self.source_images.setCurrentRow(self.project.active_source_index)
        self.source_images.blockSignals(False)
        self.remove_source_button.setEnabled(bool(self.project and len(self.project.sources)>1))
    def select_source_image(self,index):
        if not self.project or not 0<=index<len(self.project.sources):return
        self.project.select_source(index);self.hovered_segment=-1;self._populate_segments();self.refresh();self.statusBar().showMessage(f"Editing {self.project.active_source.name} — segments shown only for this image")
    def open_project(self):
        path,_=QFileDialog.getOpenFileName(self,"Open project","","Lightbox project (*.hbox)")
        if path:
            try:project=Project.load(path);self.project_path=Path(path);self.set_project(project);self.statusBar().showMessage(f"Loaded {Path(path).name}");self._schedule_session_save()
            except Exception as exc:QMessageBox.critical(self,"Unable to load project",str(exc))
    def set_project(self,project,reset_history=True):
        if reset_history:self.undo_stack.clear();self.redo_stack.clear()
        self.canvas.set_crop_mode(False)
        # Rebuilding the UI for an edit (including undo/redo) must not disturb
        # the user's viewport.  Only a genuinely new or loaded project should
        # reset zoom and pan.
        if reset_history:self.canvas.reset_zoom()
        self.crop_button.setText("Crop")
        self.project=project; self.connector_cache.invalidate(); self.hovered_segment=-1; self.layers.clear(); self.segments.clear(); blocked=(self.mode,self.min_t,self.max_t,self.mu,self.gamma,self.width_mm,self.mesh_res,self.print_smoothing,self.structure,self.backing,self.frame_width,self.frame_thickness)
        self._populate_source_images()
        for widget in blocked:widget.blockSignals(True)
        layer=project.layers[0];self.mode.setCurrentText(layer.mode.value.title());self.min_t.setValue(layer.material.minimum_mm);self.max_t.setValue(layer.material.maximum_mm);self.mu.setValue(layer.material.attenuation_per_mm);self.gamma.setValue(layer.material.gamma)
        self.width_mm.setValue(project.width_mm); self.mesh_res.setValue(project.mesh_resolution_mm); self.print_smoothing.setValue(project.print_smoothing_mm); self.structure.setCurrentIndex({"hidden_bridges":0,"backing":1,"bridges":2}.get(project.structure_mode,0)); self.backing.setValue(project.backing_thickness_mm);self.frame_width.setValue(project.frame_width_mm);self.frame_thickness.setValue(project.frame_thickness_mm)
        for widget in blocked:widget.blockSignals(False)
        for i,layer in enumerate(project.layers):
            item=QListWidgetItem(f"{layer.name}  ·  {layer.mode.value.title()}");item.setData(Qt.UserRole+1,layer.fuse_with_next);item.setData(Qt.UserRole+2,i<len(project.layers)-1);item.setToolTip("Click the link between cards to merge or separate physical output layers");item.setTextAlignment(Qt.AlignHCenter);item.setSizeHint(QSize(134,142));self.layers.addItem(item)
        add_item=QListWidgetItem(QIcon(array_pixmap(checkerboard((102,112)))),"＋  Add layer");add_item.setData(Qt.UserRole,"add");add_item.setTextAlignment(Qt.AlignHCenter);add_item.setSizeHint(QSize(134,142));add_item.setFlags(Qt.ItemIsEnabled);self.layers.addItem(add_item)
        self._populate_segments()
        self.layers.setCurrentRow(0);self.canvas_stack.setCurrentIndex(1); self._set_enabled(True); self.refresh();self._update_history_actions()
    def add_layer(self):
        if not self.project:return
        self.record_history();self.project.add_blank_layer();new_index=len(self.project.layers)-1
        self.set_project(self.project,False);self.layers.setCurrentRow(new_index);self.statusBar().showMessage(f"Added blank Layer {new_index+1}");self._schedule_session_save()
    def remove_layer(self,index):
        if not self.project:return
        if len(self.project.layers)<=2:
            self.statusBar().showMessage("A project must keep at least two layers");return
        self.record_history();removed=self.project.layers[index].name;self.project.remove_layer(index)
        next_index=min(index,len(self.project.layers)-1);self.set_project(self.project,False);self.layers.setCurrentRow(next_index);self.statusBar().showMessage(f"Removed {removed}");self._schedule_session_save()
    def move_layer(self,source,destination):
        if not self.project or source==destination:return
        self.record_history();self.project.move_layer(source,destination)
        self.set_project(self.project,False)
        self.layers.setCurrentRow(destination);self.statusBar().showMessage(f"Moved layer to position {destination+1}");self._schedule_session_save()
    def toggle_layer_link(self,index):
        if not self.project or not 0<=index<len(self.project.layers)-1:return
        self.record_history();linked=self.project.toggle_layer_link(index);item=self.layers.item(index);item.setData(Qt.UserRole+1,linked);self.layers.viewport().update();self.connector_cache.invalidate();self.refresh();self._schedule_session_save()
        groups=len(self.project.physical_layer_groups());self.statusBar().showMessage(("Linked" if linked else "Unlinked")+f" Layers {index+1} and {index+2} · {groups} physical layer(s) will export")
    def save_project(self):
        if not self.project:return
        if self.project_path:self._save_to(self.project_path)
        else:self.save_project_as()
    def save_project_as(self):
        if not self.project:return
        path,_=QFileDialog.getSaveFileName(self,"Save project","project.hbox","Lightbox project (*.hbox)")
        if path and self._save_to(Path(path)):self.project_path=Path(path)
    def _save_to(self,path):
        try:self.project.save(path);self.statusBar().showMessage(f"Saved {path}");return True
        except Exception as exc:QMessageBox.critical(self,"Unable to save project",str(exc));return False
    def reset_workspace(self):
        if self.project and QMessageBox.question(self,"Reset workspace","Clear the current project and start with an empty workspace?",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:return
        self.autosave_timer.stop();self.canvas.set_crop_mode(False);self.crop_button.setText("Crop");self.project=None;self.project_path=None;self.connector_cache.invalidate();self.hovered_segment=-1
        self.undo_stack.clear();self.redo_stack.clear();self.layers.clear();self.segments.clear();self.canvas.clear();self.illustration.clear();self.layer_heading.setText("LAYERS");self.canvas_stack.setCurrentIndex(0);self.stage_title.setText("Canvas");self.stage_meta.setText("No image loaded");self.selected_name.setText("Selected layer");self._set_enabled(False)
        try:self.session_path.unlink(missing_ok=True)
        except OSError:pass
        self.statusBar().showMessage("Workspace reset — choose an image to create a new project")
    def _schedule_session_save(self):
        if self.project:self.autosave_timer.start()
    def _save_session(self):
        if not self.project:return
        try:self.session_path.parent.mkdir(parents=True,exist_ok=True);self.project.save(self.session_path)
        except Exception as exc:self.statusBar().showMessage(f"Could not save recovery state: {exc}")
    def _restore_session(self):
        if not self.session_path.exists():return
        try:self.set_project(Project.load(self.session_path));self.statusBar().showMessage("Previous workspace restored")
        except Exception:self.statusBar().showMessage("Previous workspace could not be restored")
    def closeEvent(self,event):
        self.autosave_timer.stop();self._save_session();super().closeEvent(event)
    def export_stls(self):
        if not self.project:return
        dimensions=ExportDimensionsDialog(self)
        if dimensions.exec()!=QDialog.Accepted:return
        path=QFileDialog.getExistingDirectory(self,"Export aligned STL layers")
        if path:
            try: paths=export(self.project,Path(path),dimensions_mm=dimensions.dimensions()); QMessageBox.information(self,"Export complete",f"Exported {len(paths)} STL files and model.3mf at {dimensions.width.value():g} × {dimensions.height.value():g} × {dimensions.depth.value():g} mm, plus a backlit preview and project.hbox.")
            except Exception as exc: QMessageBox.critical(self,"Export failed",str(exc))
    def refresh_controls(self,row):
        # The layer strip contains a final "Add layer" tile.  It can briefly
        # become the current row while other controls (notably AI segment
        # results) emit refresh signals, but it is not backed by a Layer.
        if not self.project or not 0<=row<len(self.project.layers):return
        self._update_layer_previews()
        layer=self.project.layers[row]
        if not self.apply_settings_to_all.isChecked():
            self._load_layer_settings(layer)
        self.selected_name.setText("All layers" if self.apply_settings_to_all.isChecked() else layer.name)
        for widget,value in ((self.transform_x,layer.offset_x),(self.transform_y,layer.offset_y),(self.transform_scale_x,layer.output_scale_x),(self.transform_scale_y,layer.output_scale_y),(self.transform_rotation,layer.rotation_degrees)):
            widget.blockSignals(True);widget.setValue(value);widget.blockSignals(False)
        if self.project.segment_masks:self.view_box.setCurrentText(automatic_view(self.view,"Layer segments"))
        self.refresh()

    def current_layer_index(self):
        """Return a safe model index even when the add-layer tile is current."""
        if not self.project or not self.project.layers:return None
        row=self.layers.currentRow()
        return row if 0<=row<len(self.project.layers) else 0
    def _load_layer_settings(self,layer):
        values=((self.mode,layer.mode.value.title()),(self.min_t,layer.material.minimum_mm),(self.max_t,layer.material.maximum_mm),(self.mu,layer.material.attenuation_per_mm),(self.gamma,layer.material.gamma))
        for widget,value in values:
            widget.blockSignals(True)
            widget.setCurrentText(value) if widget is self.mode else widget.setValue(value)
            widget.blockSignals(False)
    def settings_changed(self,*_):
        if not self.project:return
        connector_settings=(self.project.width_mm,self.project.structure_mode,self.project.frame_width_mm)
        mode=LayerMode(self.mode.currentText().lower());minimum=self.min_t.value();maximum=max(self.max_t.value(),minimum)
        material=copy.deepcopy(self.project.layers[self.current_layer_index()].material);material.minimum_mm=minimum;material.maximum_mm=maximum;material.attenuation_per_mm=self.mu.value();material.gamma=self.gamma.value()
        indices=self.project.update_layer_settings(self.current_layer_index(),mode,material,self.apply_settings_to_all.isChecked())
        for i in indices:
            layer=self.project.layers[i];self.layers.item(i).setText(f"{layer.name}  ·  {layer.mode.value.title()}")
        self.project.width_mm=self.width_mm.value();self.project.mesh_resolution_mm=self.mesh_res.value();self.project.print_smoothing_mm=self.print_smoothing.value();self.project.structure_mode=("hidden_bridges","backing","bridges")[self.structure.currentIndex()];self.project.backing_thickness_mm=self.backing.value();self.project.frame_width_mm=self.frame_width.value();self.project.frame_thickness_mm=self.frame_thickness.value()
        if connector_settings!=(self.project.width_mm,self.project.structure_mode,self.project.frame_width_mm):self.connector_cache.invalidate()
        self.refresh();self._schedule_session_save()
    def transform_changed(self,*_):
        index=self.current_layer_index()
        if self.project is None or index is None:return
        layer=self.project.layers[index];layer.offset_x=self.transform_x.value();layer.offset_y=self.transform_y.value();layer.scale_x=self.transform_scale_x.value();layer.scale_y=self.transform_scale_y.value();layer.rotation_degrees=self.transform_rotation.value();self.connector_cache.invalidate();self._update_layer_previews();self.refresh();self._schedule_session_save()
    def mouse_transform_layer(self,mode,dx,dy):
        index=self.current_layer_index()
        if self.project is None or index is None:return
        layer=self.project.layers[index];height,width=self.project.image.shape[:2]
        self.project.transform_layer(index,mode,dx if mode=="rotate" else dx*width,dy if mode=="rotate" else dy*height)
        self.connector_cache.invalidate()
        for widget,value in ((self.transform_x,layer.offset_x),(self.transform_y,layer.offset_y),(self.transform_scale_x,layer.output_scale_x),(self.transform_scale_y,layer.output_scale_y),(self.transform_rotation,layer.rotation_degrees)):
            widget.blockSignals(True);widget.setValue(value);widget.blockSignals(False)
        self._update_layer_previews();self.refresh();self._schedule_session_save()
    def set_view(self,view):self.view=view;self.refresh()
    def add_connectors(self):
        if not self.project:return
        overlays=self.connector_cache.get(self.project)
        count=sum(int(mask.sum()) for mask in overlays)
        self.view_box.setCurrentText("Connector view")
        self.statusBar().showMessage(f"Generated {count:,} connector and frame pixels for preview; export will include them")
    def tool_changed(self,*_):
        tool=self.tool_box.currentText();wand=tool=="Magic wand";transform=tool=="Transform";pen=tool=="Pen";self.canvas.set_tool_mode("transform" if transform else ("pen" if pen else ("wand" if wand else "brush")),self.wand_inverse.isChecked())
        self.wand_inverse.setVisible(wand);self.brush.setVisible(not transform);self.brush_value.setVisible(not transform);self.brush.setRange(0 if wand else 1,100 if wand else 30);self.brush.setValue(20 if wand else 5)
        self.brush.setToolTip("Color similarity sensitivity" if wand else ("Output pen radius in pixels" if pen else "Brush radius in pixels"));self.tool_hint.setText("Draw independent output material · Right-drag to erase pen strokes" if pen else ("Drag to move · Corner to resize · Round handle to rotate" if transform else ("Click a connected color area to deselect" if wand and self.wand_inverse.isChecked() else ("Click a connected color area to select" if wand else "Drag to paint  ·  Right-drag to erase"))));self.tool_value_changed(self.brush.value())
        if transform or pen:
            if self.view!="Selected mask":self.pre_transform_view=self.view
            self.view_box.setCurrentText(view_for_tool(self.view,self.pre_transform_view,tool))
        else:
            target=view_for_tool(self.view,self.pre_transform_view,tool)
            if target=="Image" and self.project:
                layer_index=self.current_layer_index()
                if layer_index is not None:
                    source_index=self.project.layers[layer_index].source_index
                    if source_index!=self.project.active_source_index:
                        self.source_images.setCurrentRow(source_index)
            if target!=self.view:self.view_box.setCurrentText(target)
        self.refresh()
    def inverse_changed(self,checked):
        self.canvas.set_tool_mode("wand",checked);self.tool_hint.setText("Click a connected color area to deselect" if checked else "Click a connected color area to select")
    def tool_value_changed(self,value):
        self.brush_value.setText(f"{value}%" if self.tool_box.currentIndex()==1 else f"{value} px")
        if self.tool_box.currentText() in ("Brush","Pen"):self.canvas.set_brush_radius(value)
    def subtract_layer(self):
        target=self.current_layer_index()
        if not self.project or target is None:return
        choices=[(i,layer.name) for i,layer in enumerate(self.project.layers) if i!=target]
        if not choices:return
        labels=[name for _index,name in choices]
        selected,accepted=QInputDialog.getItem(self,"Subtract layer",f"Choose the layer to subtract from {self.project.layers[target].name}.",labels,0,False)
        if not accepted:return
        source=next(index for index,name in choices if name==selected)
        self.record_history();removed=self.project.subtract_layer(target,source)
        self.connector_cache.invalidate()
        self.view_box.setCurrentText("Image");self._update_layer_previews();self.refresh();self._schedule_session_save()
        self.statusBar().showMessage(f"Subtracted {self.project.layers[source].name} from {self.project.layers[target].name} · removed {removed:,} pixels")
    def add_selected_layer(self):
        target=self.current_layer_index()
        if not self.project or target is None:return
        choices=[(i,f"Layer {i+1}: {layer.name}") for i,layer in enumerate(self.project.layers) if i!=target]
        if not choices:return
        labels=[label for _index,label in choices]
        selected,accepted=QInputDialog.getItem(self,"Add layer",f"Choose the layer to add to {self.project.layers[target].name}.",labels,0,False)
        if not accepted:return
        source=next(index for index,label in choices if label==selected)
        self.record_history();added=self.project.add_layer(target,source)
        self.connector_cache.invalidate()
        self.view_box.setCurrentText("Image");self._update_layer_previews();self.refresh();self._schedule_session_save()
        self.statusBar().showMessage(f"Added {self.project.layers[source].name} to {self.project.layers[target].name} · added {added:,} pixels")
    def fill_gaps(self):
        target=self.current_layer_index()
        if not self.project or target is None:return
        overlap,accepted=QInputDialog.getInt(self,"Fill gaps",f"How many extra pixels should {self.project.layers[target].name} overlap the previous layers?",1,0,1000,1)
        if not accepted:return
        self.record_history();covered=self.project.fill_uncovered(target,overlap)
        self.connector_cache.invalidate()
        self.view_box.setCurrentText("Image");self._update_layer_previews();self.refresh();self._schedule_session_save()
        self.statusBar().showMessage(f"Filled {self.project.layers[target].name} with {covered:,} pixels · {overlap} px overlap around gaps")
    def begin_crop(self):
        enabled=not self.canvas.crop_mode;self.canvas.set_crop_mode(enabled)
        self.crop_button.setText("Cancel crop" if enabled else "Crop")
        if enabled and self.view!="Image":self.view_box.setCurrentText("Image")
        self.statusBar().showMessage(f"Drag a crop rectangle over {self.project.active_source.name}" if enabled and self.project else "Crop cancelled")
    def apply_crop(self,ax,ay,bx,by):
        self.crop_button.setText("Crop")
        if not self.project:return
        height,width=self.project.image.shape[:2]
        left=int(np.floor(min(ax,bx)*width/10000));right=int(np.ceil(max(ax,bx)*width/10000))
        top=int(np.floor(min(ay,by)*height/10000));bottom=int(np.ceil(max(ay,by)*height/10000))
        self.record_history()
        try:self.project.crop(left,top,right,bottom)
        except ValueError as exc:self.statusBar().showMessage(str(exc));return
        self.connector_cache.invalidate()
        self.hovered_segment=-1;self._populate_segments()
        self.width_mm.blockSignals(True);self.width_mm.setValue(self.project.width_mm);self.width_mm.blockSignals(False)
        layer_index=self.current_layer_index()
        if layer_index is not None:
            layer=self.project.layers[layer_index]
            for widget,value in ((self.transform_x,layer.offset_x),(self.transform_y,layer.offset_y),(self.transform_scale_x,layer.output_scale_x),(self.transform_scale_y,layer.output_scale_y),(self.transform_rotation,layer.rotation_degrees)):
                widget.blockSignals(True);widget.setValue(value);widget.blockSignals(False)
        self.refresh();self._schedule_session_save()
        height,width=self.project.image.shape[:2]
        self.statusBar().showMessage(f"Cropped {self.project.active_source.name}; other source images were unchanged")
    def _populate_segments(self):
        self.segments.blockSignals(True);self.segments.clear()
        if self.project:
            for i,mask in enumerate(self.project.segment_masks):
                item=QListWidgetItem(dot_icon(COLORS[i%len(COLORS)]),f"Segment {i+1}  ·  {int(mask.sum()):,} px");item.setSizeHint(QSize(0,42));self.segments.addItem(item)
            if self.project.segment_masks:self.segments.setCurrentRow(0)
        self.segments.blockSignals(False)
        self._update_segment_actions()
    def _update_segment_actions(self):
        enabled=bool(self.project and self.project.segment_masks and self.segments.selectedItems())
        for button in (self.replace_segment,self.add_segment,self.subtract_segment):button.setEnabled(enabled)
        self.remove_thin_button.setEnabled(bool(self.project))
        self.auto_layer_button.setEnabled(bool(self.project and self.project.segment_masks and self.project.depth_map is not None))
    def selected_segment_indices(self):
        count=len(self.project.segment_masks) if self.project else 0
        return valid_segment_indices((self.segments.row(item) for item in self.segments.selectedItems()),count)
    def select_segment(self,row):
        self._update_segment_actions()
        if self.project and 0<=row<len(self.project.segment_masks):
            self.view_box.setCurrentText(automatic_view(self.view,"Selected segment"));self.refresh()
    def segment_selection_changed(self):
        self._update_segment_actions()
        if self.segments.selectedItems():self.view_box.setCurrentText(automatic_view(self.view,"Selected segment"))
        self.refresh()
    def remove_segment(self,index):
        if not self.project or not 0<=index<len(self.project.segment_masks):return
        self.record_history()
        self.project.segment_masks.pop(index);self.hovered_segment=-1;self._populate_segments()
        if not self.project.segment_masks and self.view in ("Segmentations","Selected segment","Layer segments"):
            self.view_box.setCurrentText("Image")
        self.refresh();self._schedule_session_save()
        self.statusBar().showMessage("Segment removed — it will no longer participate in automatic layer assignment")
    def hover_segment(self,xi,yi):
        if not self.project or not self.project.segment_masks:return
        h,w=self.project.image.shape[:2];x=min(int(xi/10000*w),w-1);y=min(int(yi/10000*h),h-1)
        hits=[(int(mask.sum()),i) for i,mask in enumerate(self.project.segment_masks) if mask[y,x]]
        selected=self.segments.currentRow()
        hovered=selected if selected>=0 and any(i==selected for _area,i in hits) else (min(hits)[1] if hits else -1)
        if hovered!=self.hovered_segment:self.hovered_segment=hovered;self.refresh()
    def hover_segment_index(self,index):
        if not self.project:return
        hovered=index if 0<=index<len(self.project.segment_masks) else -1
        if hovered!=self.hovered_segment:self.hovered_segment=hovered;self.refresh()
    def clear_hover(self):
        if self.hovered_segment>=0:self.hovered_segment=-1;self.refresh()
    def select_segment_at(self,xi,yi,mode):
        if not self.project or not self.project.segment_masks:return
        h,w=self.project.image.shape[:2];x=min(int(xi/10000*w),w-1);y=min(int(yi/10000*h),h-1)
        hits=[(int(mask.sum()),i) for i,mask in enumerate(self.project.segment_masks) if mask[y,x]]
        if not hits:self.statusBar().showMessage("No segment at that point");return
        self._select_segment_index(min(hits)[1],mode)
    def _select_segment_index(self,index,mode="replace"):
        item=self.segments.item(index)
        if item is None:return
        if mode=="replace":self.segments.clearSelection();item.setSelected(True)
        elif mode=="add":item.setSelected(True)
        else:item.setSelected(False)
        self.segments.setCurrentItem(item,QItemSelectionModel.NoUpdate)
        self.segments.scrollToItem(item);self.segments.setFocus(Qt.MouseFocusReason)
        action={"replace":"Selected","add":"Added","remove":"Removed"}[mode]
        self.statusBar().showMessage(f"{action} Segment {index+1} · {len(self.segments.selectedItems())} selected")
    def show_segment_picker(self,xi,yi,global_position):
        if not self.project or not self.project.segment_masks:return
        h,w=self.project.image.shape[:2];x=min(int(xi/10000*w),w-1);y=min(int(yi/10000*h),h-1)
        hits=sorted((int(mask.sum()),i) for i,mask in enumerate(self.project.segment_masks) if mask[y,x])
        if not hits:self.statusBar().showMessage("No segment at that point");return
        menu=QMenu(self);heading=menu.addAction(f"Segments here ({len(hits)})");heading.setEnabled(False);menu.addSeparator()
        for area,index in hits:
            action=menu.addAction(dot_icon(COLORS[index%len(COLORS)]),f"Segment {index+1}  ·  {area:,} px")
            action.setData(index)
        menu.hovered.connect(lambda action:self.hover_segment_index(action.data() if action.data() is not None else -1))
        chosen=menu.exec(global_position+QPoint(12,12));self.clear_hover()
        if chosen is not None and chosen.data() is not None:self._select_segment_index(int(chosen.data()),"replace")
    def apply_segment(self,operation):
        indices=self.selected_segment_indices();li=self.layers.currentRow()
        if not self.project or not indices or li<0:return
        self.record_history();combined=np.logical_or.reduce([self.project.segment_masks[i] for i in indices]);self.project.prepare_layer_for_source(li,self.project.active_source_index);layer=self.project.layers[li]
        if operation=="replace":layer.mask=combined.copy()
        elif operation=="add":layer.mask|=combined
        else:layer.mask&=~combined
        self.connector_cache.invalidate()
        self.view_box.setCurrentText("Selected mask");self.statusBar().showMessage(f"{len(indices)} segment(s) applied to {layer.name}");self._update_layer_previews();self.refresh();self._schedule_session_save()
    def remove_thin_all_masks(self):
        if not self.project:return
        width=self.minimum_print_width.value()
        original=copy.deepcopy(self.project)
        removed=self.project.remove_thin_features_from_all_masks(width)
        if not removed:
            self.statusBar().showMessage(f"No pieces below {width:g} mm were found in any mask");return
        self.undo_stack.append(original)
        if len(self.undo_stack)>self.history_limit:self.undo_stack.pop(0)
        self.redo_stack.clear();self._update_history_actions()
        self.connector_cache.invalidate()
        for i,mask in enumerate(self.project.segment_masks):
            item=self.segments.item(i)
            if item is not None:item.setText(f"Segment {i+1}  ·  {int(mask.sum()):,} px")
        self._update_layer_previews();self.refresh();self._schedule_session_save()
        count=len(self.project.layers)+len(self.project.segment_masks)
        self.statusBar().showMessage(f"Cleaned all {count} masks · removed {removed:,} pixels below {width:g} mm")
    def clear_layer(self):
        li=self.layers.currentRow()
        if not self.project or li<0:return
        layer=self.project.layers[li]
        answer=QMessageBox.question(self,"Clear layer",f"Remove all mask content from {layer.name}?",QMessageBox.Yes|QMessageBox.Cancel,QMessageBox.Cancel)
        if answer!=QMessageBox.Yes:return
        self.record_history()
        layer.mask[:]=False
        if layer.output_additions is not None:layer.output_additions[:]=False
        if layer.output_subtractions is not None:layer.output_subtractions[:]=False
        self.connector_cache.invalidate();self.view_box.setCurrentText("Selected mask");self._update_layer_previews();self.refresh();self._schedule_session_save();self.statusBar().showMessage(f"Cleared {layer.name}")
    def full_mask(self):
        li=self.layers.currentRow()
        if not self.project or li<0:return
        self.record_history();added=self.project.fill_layer_mask(li)
        layer=self.project.layers[li]
        self.connector_cache.invalidate();self.view_box.setCurrentText("Selected mask");self._update_layer_previews();self.refresh();self._schedule_session_save();self.statusBar().showMessage(f"Selected the entire image in {layer.name} · added {added:,} pixels")
    def auto_assign_layers(self):
        if not self.project or self.project.depth_map is None or not self.project.segment_masks:return
        self.record_history()
        from .ai import assign_depth_layers, apply_ai_decomposition
        masks=assign_depth_layers(self.project.depth_map,self.project.segment_masks,len(self.project.layers))
        apply_ai_decomposition(self.project,self.project.depth_map,masks);self.connector_cache.invalidate();self._update_layer_previews();self.view_box.setCurrentText("Layer segments");self.refresh()
        self.statusBar().showMessage("Segments automatically assigned to layers by relative depth");self._schedule_session_save()
    def _update_layer_previews(self):
        if not self.project:return
        count=len(self.project.layers);current=max(0,self.layers.currentRow());self.layer_heading.setText(f"LAYERS ({current+1}/{count})")
        composite=checkerboard(self.project.image.shape[:2])
        for i in reversed(range(count)):
            layer=self.project.layers[i];color=LAYER_PREVIEW_COLORS[i%len(LAYER_PREVIEW_COLORS)];output_mask=self.project.output_mask(i)
            if i<self.layers.count():self.layers.item(i).setIcon(QIcon(array_pixmap(layer_preview(output_mask,color))))
            composite[output_mask]=QColor(color).getRgb()[:3]
        self.illustration.set_source_pixmap(array_pixmap(composite))
    def refresh(self):
        if not self.project:return
        layer_index=self.current_layer_index()
        if self.view=="Image":
            array=self.project.adjusted_image();title="Source image"
            if layer_index is not None:
                layer=self.project.layers[layer_index]
                if layer.source_index==self.project.active_source_index:
                    array=source_image_preview(array,layer.mask,layer.source_index,self.project.active_source_index);title=f"{self.project.active_source.name} — editing {layer.name}"
                else:title=f"{self.project.active_source.name} — no layer overlay"
        elif self.view=="Segmentations":
            array=self.project.adjusted_image().astype(np.float32)*.35
            for i,mask in enumerate(self.project.segment_masks):array[mask]=array[mask]*.35+np.array(QColor(COLORS[i%len(COLORS)]).getRgb()[:3])*.65
            array=np.clip(array,0,255).astype(np.uint8);title=f"{len(self.project.segment_masks)} object segmentations"
        elif self.view=="Selected segment":
            array=self.project.adjusted_image().astype(np.float32)*.35;indices=self.selected_segment_indices()
            for i in indices:
                mask=self.project.segment_masks[i];color=np.array(QColor(COLORS[i%len(COLORS)]).getRgb()[:3]);array[mask]=array[mask]*.35+color*.65
            array=np.clip(array,0,255).astype(np.uint8);title=f"{len(indices)} selected segment(s)" if indices else "No segment selected"
        elif self.view=="Layer segments":
            if layer_index is None:return
            li=layer_index;layer_mask=self.project.layers[li].mask
            array=self.project.adjusted_image().astype(np.float32)*.25
            covered=np.zeros(layer_mask.shape,bool)
            for i,segment in enumerate(self.project.segment_masks):
                region=segment&layer_mask;covered|=region
                color=np.array(QColor(COLORS[i%len(COLORS)]).getRgb()[:3]);array[region]=array[region]*.25+color*.75
            remainder=layer_mask&~covered;array[remainder]=array[remainder]*.25+np.array([139,108,255])*.75
            array=np.clip(array,0,255).astype(np.uint8);title=f"{self.project.layers[li].name} — constituent segments"
        elif self.view=="Connector view":
            if layer_index is None:return
            connectors=self.connector_cache.get(self.project)[layer_index]
            array=connector_view_preview(self.project.output_image(layer_index),self.project.output_mask(layer_index),connectors)
            title=f"{self.project.layers[layer_index].name} — {int(connectors.sum()):,} connector and frame pixels"
        elif self.view=="Depth map":
            if self.project.depth_map is None: array=np.zeros(self.project.image.shape[:2],np.uint8);title="Depth map — run Auto Layers with AI"
            else: array=np.clip(self.project.depth_map*255,0,255).astype(np.uint8);title="AI relative depth — brighter is nearer"
        elif self.view=="Backlit preview":array=preview_u8(self.project);title="Simulated backlight"
        else:
            if layer_index is None:return
            i=layer_index;mask=self.project.output_mask(i);array=np.zeros((*mask.shape,3),np.uint8);array[:]=[18,21,27];array[mask]=[139,108,255];title=f"{self.project.layers[i].name} output mask"
        if 0<=self.hovered_segment<len(self.project.segment_masks):
            mask=self.project.segment_masks[self.hovered_segment]
            array=np.asarray(array)
            if array.ndim==2:array=np.repeat(array[...,None],3,axis=2)
            else:array=array.copy()
            array[mask]=np.clip(array[mask].astype(np.float32)*.45+np.array([255,255,255])*.55,0,255).astype(np.uint8);array[mask_outline(mask)]=[255,255,255]
            title=f"Segment {self.hovered_segment+1} highlighted"
        h,w=self.project.image.shape[:2]
        bounds=None
        if layer_index is not None:
            ys,xs=np.where(self.project.output_mask(layer_index))
            if len(xs):bounds=(xs.min()/w,ys.min()/h,(xs.max()-xs.min()+1)/w,(ys.max()-ys.min()+1)/h)
        self.canvas.set_transform_bounds(bounds);self.stage_title.setText(title);self.stage_meta.setText(f"{w} × {h} px  ·  {self.project.width_mm:.0f} × {self.project.height_mm:.0f} mm");self.canvas.set_source_pixmap(array_pixmap(array))
    def resizeEvent(self,event):super().resizeEvent(event);self.refresh()
    def paint_mask(self,xi,yi,erase):
        if not self.project:return
        h,w=self.project.image.shape[:2];x=min(int(xi/10000*w),w-1);y=min(int(yi/10000*h),h-1)
        if self.tool_box.currentText()=="Pen":
            layer_index=self.current_layer_index()
            if layer_index is None:return
            layer=self.project.layers[layer_index]
            if layer.output_additions is None:layer.output_additions=np.zeros((h,w),bool)
            r=self.brush.value();yy,xx=np.ogrid[:h,:w];disk=(xx-x)**2+(yy-y)**2<=r*r;layer.output_additions[disk]=not erase
            self.connector_cache.invalidate();self._update_layer_previews();self.refresh();self._schedule_session_save();return
        if self.view in ("Segmentations","Layer segments"):
            layer_mask=self.project.layers[max(0,self.layers.currentRow())].mask if self.view=="Layer segments" else None
            hits=[(int(mask.sum()),i) for i,mask in enumerate(self.project.segment_masks) if mask[y,x] and (layer_mask is None or layer_mask[y,x])]
            if hits:self.segments.setCurrentRow(min(hits)[1])
            else:self.statusBar().showMessage("No segment at that point — choose one from the segment list")
            return
        r=self.brush.value();yy,xx=np.ogrid[:h,:w];disk=(xx-x)**2+(yy-y)**2<=r*r
        if self.view=="Selected segment" and self.segments.currentRow()>=0:
            mask=self.project.segment_masks[self.segments.currentRow()];mask[disk]=not erase
            self.segments.currentItem().setText(f"Segment {self.segments.currentRow()+1}  ·  {int(mask.sum()):,} px")
        else:
            layer_index=self.current_layer_index()
            if layer_index is None:return
            self.project.prepare_layer_for_source(layer_index,self.project.active_source_index);self.project.layers[layer_index].mask[disk]=not erase;self.connector_cache.invalidate();self._update_layer_previews()
        self.refresh();self._schedule_session_save()
    def apply_wand(self,xi,yi,inverse):
        if not self.project:return
        h,w=self.project.image.shape[:2];x=min(int(xi/10000*w),w-1);y=min(int(yi/10000*h),h-1)
        region=magic_wand_region(self.project.adjusted_image(),x,y,self.brush.value());self.record_history()
        if self.view=="Selected segment" and self.segments.currentRow()>=0:
            mask=self.project.segment_masks[self.segments.currentRow()];mask[region]=not inverse
            self.segments.currentItem().setText(f"Segment {self.segments.currentRow()+1}  ·  {int(mask.sum()):,} px")
            target=f"Segment {self.segments.currentRow()+1}"
        else:
            row=self.current_layer_index()
            if row is None:return
            self.project.prepare_layer_for_source(row,self.project.active_source_index);mask=self.project.layers[row].mask;mask[region]=not inverse;self.connector_cache.invalidate();target=self.project.layers[row].name;self._update_layer_previews()
        action="Removed" if inverse else "Selected";self.statusBar().showMessage(f"{action} {int(region.sum()):,} connected pixels in {target}");self.refresh();self._schedule_session_save()
    def run_optimize(self):
        if self.project:
            self.record_history();count=optimize_rear_openings(self.project,max(0,self.layers.currentRow()));self.connector_cache.invalidate();self._update_layer_previews();self.view_box.setCurrentText("Backlit preview");self.statusBar().showMessage(f"Opened {count:,} rear-layer pixels");self._schedule_session_save()
    def run_ai(self):
        if not self.project:return
        self.ai_button.setEnabled(False);self.ai_button.setText("Loading local models…")
        quality=("accurate","base","fast")[self.ai_quality.currentIndex()]
        self._ai_thread=QThread(self);self._ai_worker=AIWorker(self.project.image.copy(),len(self.project.layers),None,quality,self.minimum_segment_area.value());self._ai_worker.moveToThread(self._ai_thread);self._ai_thread.started.connect(self._ai_worker.run);self._ai_worker.progress.connect(self._ai_progress);self._ai_worker.finished.connect(self._ai_finished);self._ai_worker.failed.connect(self._ai_failed);self._ai_worker.finished.connect(self._ai_thread.quit);self._ai_worker.failed.connect(self._ai_thread.quit);self._ai_thread.start()
    def _ai_progress(self,message):self.statusBar().showMessage(message);self.ai_button.setText(message.replace("…",""))
    def _ai_finished(self,depth,segments):
        self.record_history();self.project.depth_map=np.asarray(depth,dtype=np.float32);self.project.segment_masks=[np.asarray(mask,dtype=bool) for mask in segments];self.project._store_active_source();self._populate_source_images();self._populate_segments();self.ai_button.setEnabled(True);self.ai_button.setText("◉  Segment image");self.view_box.setCurrentText("Segmentations");self.refresh();self.statusBar().showMessage("Segmentation complete — cached for this source image");self._schedule_session_save()
        if not segments:QMessageBox.information(self,"No segments found","No object segments passed the current quality thresholds. Try another quality setting or use the existing layer-mask brush tools.")
    def _ai_failed(self,message):
        self.ai_button.setEnabled(True);self.ai_button.setText("◉  Segment image");QMessageBox.critical(self,"Local AI unavailable",message);self.statusBar().showMessage("AI segmentation failed; manual editing is still available")
    def run_validate(self):
        if not self.project:return
        from .geometry import connected_components
        messages=[f"{layer.name}: {len(connected_components(self.project.output_mask(i)))} connected component(s)" for i,layer in enumerate(self.project.layers)];QMessageBox.information(self,"Structural validation","\n".join(messages)+"\n\nExport adds a perimeter and bridges islands. Non-watertight meshes can still be exported.")

    def record_history(self):
        if not self.project:return
        self.undo_stack.append(copy.deepcopy(self.project))
        if len(self.undo_stack)>self.history_limit:self.undo_stack.pop(0)
        self.redo_stack.clear();self._update_history_actions()
    def begin_brush_history(self):
        if self.view not in ("Segmentations","Layer segments"):
            self.record_history()
    def _restore_history(self,project,message):
        # Rebuilding the controls emits their selection signals.  In projects
        # with generated segments those signals switch the canvas to
        # "Layer segments", which makes a correctly restored brush/wand mask
        # look unchanged.  Preserve the editing context across history moves
        # and restore it after the controls have been repopulated.
        layer=max(0,self.layers.currentRow());view=self.view
        selected_segments=self.selected_segment_indices();current_segment=self.segments.currentRow()
        self.set_project(copy.deepcopy(project),False)
        self.layers.setCurrentRow(min(layer,len(self.project.layers)-1))
        if self.segments.count():
            self.segments.clearSelection()
            for index in selected_segments:
                item=self.segments.item(index)
                if item is not None:item.setSelected(True)
            current=min(max(current_segment,0),self.segments.count()-1)
            self.segments.setCurrentItem(self.segments.item(current),QItemSelectionModel.NoUpdate)
        self.view_box.setCurrentText(view)
        self.view=view;self.refresh()
        self.statusBar().showMessage(message);self._schedule_session_save();self._update_history_actions()
    def undo(self):
        if not self.project or not self.undo_stack:return
        self.redo_stack.append(copy.deepcopy(self.project));previous=self.undo_stack.pop();self._restore_history(previous,"Undid last edit")
    def redo(self):
        if not self.project or not self.redo_stack:return
        self.undo_stack.append(copy.deepcopy(self.project));next_project=self.redo_stack.pop();self._restore_history(next_project,"Redid last edit")
    def _update_history_actions(self):
        if hasattr(self,"undo_btn"):
            self.undo_btn.setEnabled(bool(self.project and self.undo_stack));self.redo_btn.setEnabled(bool(self.project and self.redo_stack))

def main():
    app=QApplication(sys.argv);app.setApplicationName("LumaLayer");window=MainWindow();window.show();sys.exit(app.exec())
if __name__=="__main__":main()
