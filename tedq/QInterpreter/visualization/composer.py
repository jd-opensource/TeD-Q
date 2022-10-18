import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np
from tedq.QInterpreter.operators.qubit import *
from tedq.QInterpreter.operators.measurement import *
from tedq.QInterpreter.circuits.circuit import *

class circuit_composer:
    def __init__(self, n_qubits, **kwargs):
        self._fig_size = kwargs.get("figsize", (12,5))
        self._dpi = kwargs.get("dpi", 50)
        self._scale = kwargs.get("scale", 1.2)
        self._fig_width = self._dpi*self._fig_size[0]
        self._fig_height = self._dpi*self._fig_size[1]
        self._qubit_num = n_qubits


        self._plt = plt
        self._fig= plt.figure(figsize=self._fig_size, dpi=self._dpi)
        self._ax = plt.gca()
        self._patch = mpatch

        # Set fig environment
        self._fig.canvas.toolbar_visible = False
        self._fig.canvas.header_visible = False
        self._fig.canvas.footer_visible = False

        # set ax environment
        self._ax.axis("off")
        self._ax.set_aspect(self._fig_height/self._fig_width)
        self._ax.tick_params(
            labelbottom=False, labeltop=False, labelleft=False, labelright=False
        )
        self._ax.set_title("")
        
        self._line_width = 2*self._scale
        self._layer_width = 0.6*self._scale
        self._line_inter_val = 0.5*self._scale
        self._basic_gate_width = 0.4*self._scale
        self._control_radius = 0.05*self._scale
        self._layer_blank_width = 0.05*self._scale
        self._reverse_qubit = False

        self._div_x = 60*self._scale/self._fig_width
        self._div_y = 60*self._scale/self._fig_height*(-1)
        self._gate_dx = 40*self._scale/self._fig_width
        self._gate_dy = 40*self._scale/self._fig_height*(-1)
        self._gate_bw = 10*self._scale/self._fig_width
        self._wire_width = 5*self._scale/self._fig_width
        self._control_radius = 10*self._scale/self._fig_width

        self._font_size = 12.*self._fig_width/720

        self._gate_template_list = []

        self.sheet = []
        for idx in range(self._qubit_num):
            self.sheet.append([])
        self.max_qubit_idx = [0]*4
        self.operators = []
        self.measurements = []
        self.obj_on_sheet = []
        self.draggable_obj_on_sheet=[]

        self._DrawTemplate()
        self._DrawMeasurementsTemplate()
        self._DrawTrashCan()
        self._plt.show()

    def _GetXYPosFromXYIndex(self, xi, yi):
        pos_x = xi*self._div_x
        pos_y = (yi+2)*self._div_y +1
        return (pos_x, pos_y)

    def _GetXYIndexFromXYPos(self, x, y):
        idx_x = int(np.round(x/self._div_x))
        idx_y = int(np.round((y-1)/self._div_y))-2
        return (idx_x, idx_y)

    def _PlotRectangle(self, x,y,w,h,zorder=2,**kwargs):
        box = self._patch.Rectangle(
            xy=(x-0.5*w, y-0.5*h),
            width=w,
            height=h,
            zorder=zorder,         #TODO
            **kwargs
        )
        self._ax.add_patch(box)
        return box

    def _PlotText(self, x,y,text,**kwargs):
        textbox = self._ax.text(
            x,
            y,
            text,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=self._font_size,
            clip_on=True,
            zorder=4,         #TODO
            **kwargs
        )
        return textbox

    def _DrawGate(self,xi, yi, text, params=None, **kwargs):
        (pos_x, pos_y) = self._GetXYPosFromXYIndex(xi, yi)
        gate_w = self._gate_dx
        gate_h = self._gate_dy


        rect = self._PlotRectangle(pos_x, pos_y, gate_w, gate_h, **kwargs)
        textbox = self._PlotText(pos_x, pos_y, text)
        

        return rect, textbox

    def _DrawQuantumWire(self, xi, yi):
        (pos_x, pos_y) = self._GetXYPosFromXYIndex(xi, yi)
        self._ax.plot(
            [pos_x - self._div_x/2, pos_x + self._div_x/2],
            [pos_y, pos_y],
            color="black",
            # linewidth=self._wire_width,
            zorder=1,
        )
    def _PlotParams(self, x,y,param,**kwargs):
        text = self._ax.text(
            x,
            y,
            "("+str(np.round(float(param),2))+")",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=self._font_size*0.5,
            clip_on=True,
            zorder=4,         #TODO
            **kwargs
        )
        self.obj_on_sheet.append(text)


    def _DrawCircuitGate(self,xi, yi, text, params=None, **kwargs):
        (pos_x, pos_y) = self._GetXYPosFromXYIndex(xi, yi)
        gate_w = self._gate_dx
        gate_h = self._gate_dy


        self._PlotRectangle(pos_x, pos_y, gate_w, gate_h, **kwargs)
        if params==None :
            self._PlotText(pos_x, pos_y, text)
        else:
            self._PlotText(pos_x, pos_y-0.1*self._gate_dy, text)
            self._PlotParams(pos_x, pos_y+0.35*self._gate_dy, params[0])

    def _DrawSingleQubitGate(self,xi, yarr, operator):
        if operator.is_preparation:
            color="darkturquoise"
        else:
            color="skyblue"
        rect, textbox = self._DrawGate(xi, yarr[0], SingleQubitGates[operator.name], operator.parameters, color=color)
        gate = DraggableSingleQubitGate(textbox, rect, self.operators.index(operator), self)
        gate.connect()
        self.obj_on_sheet.append(rect)
        self.obj_on_sheet.append(textbox)
        self.draggable_obj_on_sheet.append(gate)

    def _DrawControlledQubitGate(self,xi, yarr, operator):
        rect, textbox = self._DrawGate(xi, yarr[1], ControlledGates[operator.name], color="lightcoral")
        line = self._DrawConnectionWire(xi, yarr[0], yarr[1])
        dot = self._DrawControlQubit(xi, yarr[0])
        self.obj_on_sheet.append(rect)
        self.obj_on_sheet.append(textbox)
        gate = DraggableControlledQubitGate(textbox, rect, line[0], dot, self.operators.index(operator), self)
        gate.connect()
        self.draggable_obj_on_sheet.append(gate)
    def _DrawControlQubit(self, xi, yi):
        (pos_x, pos_y) = self._GetXYPosFromXYIndex(xi, yi)
        circle = self._patch.Ellipse(
            xy=(pos_x, pos_y), 
            width=self._gate_dx/3,
            height=self._gate_dy/3, 
            color="black", 
            zorder=3
        )
        self._ax.add_patch(circle)
        self.obj_on_sheet.append(circle)
        return circle
    def _DrawConnectionWire(self,xi, yi, yj):
        (pos_x_i, pos_y_i) = self._GetXYPosFromXYIndex(xi, yi)
        (pos_x_i, pos_y_j) = self._GetXYPosFromXYIndex(xi, yj)
        line = self._ax.plot(
            [pos_x_i, pos_x_i],
            [pos_y_i, pos_y_j],
            color="black",
            # linewidth=self._wire_width,
            zorder=1,
        ) 
        self.obj_on_sheet.append(line)
        return line

    def _DrawCircuit(self):
        layers = self._ParseCircuit()
        for i, layer in enumerate(layers):
            for op in layer:
                # for j in range(self._qubit_num):
                    # self._DrawQuantumWire(i, j)
                    # pass
                if (op.name in SingleQubitGates):
                    self._DrawSingleQubitGate(i, op.qubits, op)
                elif (op.name in ControlledGates):
                    self._DrawControlledQubitGate(i, op.qubits, op)
        self._DrawMeasurements()
                

    def _ParseCircuit(self):
        head_of_wires = [0 for i in range(self._qubit_num)]
        wires = []
        for op in self.operators:
            covered_wires = self._CoveredWires(op)
            cur_head = max([head_of_wires[i] for i in covered_wires])
            while cur_head<op._expected_index:
                cur_head+=1
                if len(wires)<op._expected_index:
                    wires.append([])
                    
            for wire in covered_wires:
                head_of_wires[wire] = cur_head+1
            if len(wires) <= cur_head:
                wires.append([op])
            else:
                wires[cur_head].append(op)
        return wires
    def _ParseCircuitHead(self):
        head_of_wires = [0 for i in range(self._qubit_num)]
        for op in self.operators:
            covered_wires = self._CoveredWires(op)
            cur_head = max([head_of_wires[i] for i in covered_wires])
            while cur_head<op._expected_index:
                cur_head+=1
                
                    
            for wire in covered_wires:
                head_of_wires[wire] = cur_head+1
        self.max_qubit_idx = head_of_wires
        return head_of_wires
    def _DrawTrashCan(self):
        rect, textbox = self._DrawGate(0, -1, "Del", zorder=3, color="lightgray")
    def _DrawTemplate(self):
        for idx in range(len(GateNameList)):
            rect, textbox = self._DrawGate(idx+2, -1, GateNameList[idx], zorder=3, color="navajowhite")
            gateTemplate = DraggableGateTemplate(textbox, rect, GateNameList[idx], self)
            gateTemplate.connect()
            self._gate_template_list.append(gateTemplate)

        for jdx in range(self._qubit_num):
            for idx in range(15):       
                self._DrawQuantumWire(idx, jdx)
    def _DrawMeasurementsTemplate(self):

        (pos_x, pos_y) = self._GetXYPosFromXYIndex(len(GateNameList)+3, -1)
        gate_w = self._gate_dx
        gate_h = self._gate_dy

        rect = self._PlotRectangle(pos_x, pos_y, gate_w, gate_h, facecolor="white", edgecolor="black")

        circle = self._patch.Arc(
            xy=(pos_x, pos_y+0.2*gate_h),
            width=-gate_w*0.8,
            height=-gate_h*0.8,
            # fc=fc,
            # ec=ec,
            # linewidth=self._lwidth15,
            zorder=6,
            theta1=0,
            theta2=180,
            color="black",
        )
        
        textbox = self._ax.text(
            pos_x - 0.35 * gate_w,
            pos_y - 0.3 * gate_h,
            SingleQubitGates["PauliZ"],
            ha="center",
            va="center",
            fontsize=10,
            # color=gt,
            clip_on=True,
            zorder=7,
        )
        self._ax.add_patch(circle)
        arrow = self._ax.arrow(pos_x, pos_y+0.2*gate_h, gate_w*0.5/2, -gate_h*0.5/2, zorder=7, width=0.002,head_width=0.006, color="black")
        gateTemplate = DraggableMeasurementTemplate(textbox, rect, arrow, circle, self)
        gateTemplate.connect()
        self._gate_template_list.append(gateTemplate)
    def _GetOpertorFromName(self,name, qubits):
        operator = GateList[name]

        if name in ["Rx", "Ry", "Rz"]:
            new_op = operator(0, qubits = qubits, do_queue=False)
        else:
            new_op = operator(qubits = qubits, do_queue=False)
        return new_op
    def _GetMeasurementFromName(self,name, qubits):
        return expval(PauliZ(qubits=qubits, do_queue=False), do_queue=False)

    def _DrawMeasurements(self, color="black"):
        measurements = self.measurements
        if measurements is None:
            return
        head_of_wires = 14
        
        for measurement in measurements:

            (pos_x, pos_y) = self._GetXYPosFromXYIndex(head_of_wires, measurement.obs.qubits[0])
            # self._DrawQuantumWire(head_of_wires-0.5, measurement.obs.qubits[0])
            gate_w = self._gate_dx
            gate_h = self._gate_dy


            rect = self._PlotRectangle(pos_x, pos_y, gate_w, gate_h, facecolor="white", edgecolor="black")

            circle = self._patch.Arc(
                xy=(pos_x, pos_y+0.2*gate_h),
                width=-gate_w*0.8,
                height=-gate_h*0.8,
                # fc=fc,
                # ec=ec,
                # linewidth=self._lwidth15,
                zorder=6,
                theta1=0,
                theta2=180,
                color="black",
            )
            
            textbox = self._ax.text(
                pos_x - 0.35 * gate_w,
                pos_y - 0.3 * gate_h,
                SingleQubitGates[measurement.obs.name],
                ha="center",
                va="center",
                fontsize=10,
                # color=gt,
                clip_on=True,
                zorder=7,
            )
            self._ax.add_patch(circle)
            arrow = self._ax.arrow(pos_x, pos_y+0.2*gate_h, gate_w*0.5/2, -gate_h*0.5/2, zorder=7, width=0.002,head_width=0.004, color="black")

            gate = DraggableMeasurement(textbox,rect, arrow, circle, self, measurement)
            gate.connect()
            self.draggable_obj_on_sheet.append(gate)
            self.obj_on_sheet.append(rect)
            self.obj_on_sheet.append(circle)
            self.obj_on_sheet.append(arrow)
            self.obj_on_sheet.append(textbox)

    def addSingleQubitGate(self, x, y, name):
        idx_x, idx_y = self._GetXYIndexFromXYPos(x, y)
        if idx_y<0 or idx_x<0:
            return
        if idx_y<self._qubit_num:
            self._ParseCircuitHead()
            if name == "CNOT":
                if idx_y<self._qubit_num and idx_y>0:
                    ops = self._GetOpertorFromName("CNOT", [idx_y-1, idx_y])
                else:
                    return
            else:
                ops = self._GetOpertorFromName(name, [idx_y])

            covered_wires = ops.qubits
            cur_head = max([self.max_qubit_idx[i] for i in covered_wires])
            if idx_x>=cur_head:
                self.operators.append(ops)
                for wire in covered_wires:
                    self.max_qubit_idx[wire] = cur_head+1
            else:
                counter = [0]*self._qubit_num
                idx = -1
                isFoundPos = False
                while not isFoundPos:
                    idx+=1
                    

                    if idx < len(self.operators):
                        qubits = self.operators[idx].qubits
                    else:
                        isFoundPos = True
                    head = max([counter[qubit] for qubit in qubits])+1
                    for qubit in qubits:
                        counter[qubit] = head

                    max_count = max([counter[wire] for wire in covered_wires])
                    if max_count>=idx_x:
                        isFoundPos = True

                self.operators.insert(idx+1, ops)




        for obj in self.draggable_obj_on_sheet:
            del obj
        for obj in self.obj_on_sheet:
            if isinstance(obj, list):
                for sub_obj in obj:
                    sub_obj.remove()
            else:
                obj.remove()
        
        self.obj_on_sheet = []
        self.draggable_obj_on_sheet = []
        self._DrawCircuit()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
    def updateSingleQubitGate(self, x, y, gate):
        idx_x, idx_y = self._GetXYIndexFromXYPos(x, y)
        if (idx_y<0 or idx_x<0) and not (idx_x == 0 and idx_y == -1):
            return
        
            
        old_ops = self.operators.pop(gate.idx)
        
        
        self._ParseCircuitHead()
        
        ops = self._GetOpertorFromName(SingleQubitGates[old_ops.name], [idx_y])
        del old_ops


        if idx_y<self._qubit_num and not (idx_x == 0 and idx_y == -1):
            covered_wires = ops.qubits
            cur_head = max([self.max_qubit_idx[i] for i in covered_wires])
            if idx_x>=cur_head:
                self.operators.append(ops)
                for wire in covered_wires:
                    self.max_qubit_idx[wire] = cur_head+1
            else:
                counter = [0]*self._qubit_num
                idx = -1
                isFoundPos = False
                while not isFoundPos:
                    idx+=1
                    

                    if idx < len(self.operators):
                        qubits = self.operators[idx].qubits
                    else:
                        isFoundPos = True
                    head = max([counter[qubit] for qubit in qubits])+1
                    for qubit in qubits:
                        counter[qubit] = head

                    for qubit, count in enumerate(counter):
                        if qubit in covered_wires and count >idx_x:
                            isFoundPos = True
                self.operators.insert(idx, ops)




        for obj in self.draggable_obj_on_sheet:
            del obj
        for obj in self.obj_on_sheet:
            if isinstance(obj, list):
                for sub_obj in obj:
                    sub_obj.remove()
            else:
                obj.remove()
        
        self.obj_on_sheet = []
        self.draggable_obj_on_sheet = []
        self._DrawCircuit()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def updateControlledQubitGate(self, x, y, gate):
        idx_x, idx_y = self._GetXYIndexFromXYPos(x, y)
        if (idx_y<0 or idx_x<0) and not (idx_x == 0 and idx_y == -1):
            return
        if idx_y==self.operators[gate.idx].qubits[0]:
            return
            
        old_ops = self.operators.pop(gate.idx)
        
        self._ParseCircuitHead()

        
        ops = self._GetOpertorFromName("CNOT", [old_ops.qubits[0], idx_y])
        del old_ops


        if idx_y<self._qubit_num and not (idx_x == 0 and idx_y == -1):            
            covered_wires = ops.qubits
            cur_head = max([self.max_qubit_idx[i] for i in covered_wires])
            if idx_x>=cur_head:
                self.operators.append(ops)
                for wire in covered_wires:
                    self.max_qubit_idx[wire] = cur_head+1
            else:
                counter = [0]*self._qubit_num
                idx = -1
                isFoundPos = False
                while not isFoundPos:
                    idx+=1
                    

                    if idx < len(self.operators):
                        qubits = self.operators[idx].qubits
                    else:
                        isFoundPos = True
                    head = max([counter[qubit] for qubit in qubits])+1
                    for qubit in qubits:
                        counter[qubit] = head

                    for qubit, count in enumerate(counter):
                        if qubit in covered_wires and count >idx_x:
                            isFoundPos = True
                self.operators.insert(idx, ops)

        for obj in self.draggable_obj_on_sheet:
            del obj
        for obj in self.obj_on_sheet:
            if isinstance(obj, list):
                for sub_obj in obj:
                    sub_obj.remove()
            else:
                obj.remove()
        
        self.obj_on_sheet = []
        self.draggable_obj_on_sheet = []
        self._DrawCircuit()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def updateControlledQubitGateController(self, x, y, gate):
        idx_x, idx_y = self._GetXYIndexFromXYPos(x, y)
        if (idx_y<0 or idx_x<0) and not (idx_x == 0 and idx_y == -1):
            return
        if idx_y==self.operators[gate.idx].qubits[1]:
            
            return
            
        old_ops = self.operators.pop(gate.idx)
        
        self._ParseCircuitHead()

        
        ops = self._GetOpertorFromName("CNOT", [idx_y, old_ops.qubits[1]])
        del old_ops


        if idx_y<self._qubit_num and not (idx_x == 0 and idx_y == -1):
            covered_wires = ops.qubits
            cur_head = max([self.max_qubit_idx[i] for i in covered_wires])
            if idx_x>=cur_head:
                self.operators.append(ops)
                for wire in covered_wires:
                    self.max_qubit_idx[wire] = cur_head+1
            else:
                counter = [0]*self._qubit_num
                idx = -1
                isFoundPos = False
                while not isFoundPos:
                    idx+=1
                    

                    if idx < len(self.operators):
                        qubits = self.operators[idx].qubits
                    else:
                        isFoundPos = True
                    head = max([counter[qubit] for qubit in qubits])+1
                    for qubit in qubits:
                        counter[qubit] = head

                    for qubit, count in enumerate(counter):
                        if qubit in covered_wires and count >idx_x:
                            isFoundPos = True
                self.operators.insert(idx, ops)

        for obj in self.draggable_obj_on_sheet:
            del obj
        for obj in self.obj_on_sheet:
            if isinstance(obj, list):
                for sub_obj in obj:
                    sub_obj.remove()
            else:
                obj.remove()
        
        self.obj_on_sheet = []
        self.draggable_obj_on_sheet = []
        self._DrawCircuit()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
    def _ParseMeasurement(self):
        counter = [0]*self._qubit_num
        for measurement in self.measurements:
            qubits = measurement.obs.qubits
            counter[qubits[0]]+=1
        return counter
    def addMeasurement(self, x, y, gate):
        idx_x, idx_y = self._GetXYIndexFromXYPos(x, y)
        counter = self._ParseMeasurement()

        if idx_y<0 or idx_x<0 or counter[idx_y]>0:
            return
        if idx_y<self._qubit_num:
            self._ParseCircuitHead()

            gate = self._GetMeasurementFromName("PZ", qubits=[idx_y])
            self.measurements.append(gate)


        for obj in self.draggable_obj_on_sheet:
            del obj
        for obj in self.obj_on_sheet:
            if isinstance(obj, list):
                for sub_obj in obj:
                    sub_obj.remove()
            else:
                obj.remove()
        
        self.obj_on_sheet = []
        self.draggable_obj_on_sheet = []
        self._DrawCircuit()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
    def updateMeasurement(self, x, y, gate):
        idx_x, idx_y = self._GetXYIndexFromXYPos(x, y)
        counter = self._ParseMeasurement()


        if (idx_y<0 or idx_x<0) and not (idx_x == 0 and idx_y == -1):
            return


        old_ops = self.measurements.pop(self.measurements.index(gate.measurement))

        if idx_y<self._qubit_num and not (idx_x == 0 and idx_y == -1):
            self._ParseCircuitHead()

            gate = self._GetMeasurementFromName("PZ", qubits=[idx_y])

            self.measurements.append(gate)



        for obj in self.draggable_obj_on_sheet:
            del obj
        for obj in self.obj_on_sheet:
            if isinstance(obj, list):
                for sub_obj in obj:
                    sub_obj.remove()
            else:
                obj.remove()
        
        self.obj_on_sheet = []
        self.draggable_obj_on_sheet = []
        self._DrawCircuit()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def toCircuit(self):
        circuit_material = {"operators": self.operators, "measurements": self.measurements, "init_state": None}
        circuit = Circuit(circuit_material, self._qubit_num)
        return circuit

    @classmethod
    def _CoveredWires(cls, operator):
        qubit_min = min(operator.qubits)
        qubit_max = max(operator.qubits)
        return [i for i in range(qubit_min, qubit_max + 1)]


class DraggableGateTemplate:
    def __init__(self, textbox, rect, name, composer):
        self.textbox = textbox
        self.name = name
        self.rect = rect
        self.press = None
        self.prev = None
        self.composer = composer
        self.x0, self.y0 = self.rect.xy
        self.h = self.rect.get_height()
        self.w = self.rect.get_width()
        self._dpi = composer._dpi 
        self._fig_x , self._fig_y = composer._fig_size

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        # self.cidmotion = self.rect.figure.canvas.mpl_connect(
        #     'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.rect.contains(event)
        if not contains: return
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        if self.prev is not None:
            xprev, yprev = self.prev
            dx = event.xdata - xprev
            dy = event.ydata - yprev
            if abs(dx + dy) < 1/self._fig_x:
                return
        
        
                
                
        self.prev = event.xdata, event.ydata
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress


        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)
        
        self.textbox.set_x(x0+dx+0.5*self.w)
        self.textbox.set_y(y0+dy+0.5*self.h)

        # self.rect.figure.canvas.draw()
        # self.textbox.figure.canvas.draw()
        self.rect.figure.canvas.draw()


    def on_release(self, event):
        'on release we reset the press data'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return

        x1, y1 = event.xdata, event.ydata
        x1, y1 = x1+ 0.5*self.w,y1+ 0.5*self.h

        self.composer.addSingleQubitGate(x1, y1, self.name)

        self.rect.set_x(self.x0)
        self.rect.set_y(self.y0)
        
        self.textbox.set_x(self.x0+0.5*self.w)
        self.textbox.set_y(self.y0+0.5*self.h)
        self.press = None
        self.rect.figure.canvas.draw()
        # self.rect.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

class DraggableSingleQubitGate:
    def __init__(self, textbox, rect, idx, composer):
        self.textbox = textbox
        self.idx = idx
        self.rect = rect
        self.press = None
        self.prev = None
        self.composer = composer
        self.x0, self.y0 = self.rect.xy
        self.h = self.rect.get_height()
        self.w = self.rect.get_width()
        self._dpi = composer._dpi 
        self._fig_x , self._fig_y = composer._fig_size

    def __del__(self):
        # self.disconnect()
        pass

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.rect.contains(event)
        if not contains: return
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        if self.prev is not None:
            xprev, yprev = self.prev
            dx = event.xdata - xprev
            dy = event.ydata - yprev
            if abs(dx + dy) < 0.5/self._fig_x:
                return
        
        
                
                
        self.prev = event.xdata, event.ydata
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress


        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)
        
        self.textbox.set_x(x0+dx+0.5*self.w)
        self.textbox.set_y(y0+dy+0.5*self.h)

        # self.rect.figure.canvas.draw()
        # self.textbox.figure.canvas.draw()
        self.rect.figure.canvas.draw()


    def on_release(self, event):
        'on release we reset the press data'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return

        x1, y1 = event.xdata, event.ydata
        x1, y1 = x1+ 0.5*self.w,y1+ 0.5*self.h

        

        self.rect.set_x(self.x0)
        self.rect.set_y(self.y0)
        
        self.textbox.set_x(self.x0+0.5*self.w)
        self.textbox.set_y(self.y0+0.5*self.h)
        self.press = None
        self.prev = None

        # self.rect.figure.canvas.draw()
        # self.rect.figure.canvas.draw()

        self.composer.updateSingleQubitGate(x1, y1, self)


    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

class DraggableControlledQubitGate:
    def __init__(self, textbox, rect, line, circle, idx, composer):
        self.textbox = textbox
        self.idx = idx
        self.rect = rect
        self.line = line 
        self.circle = circle
        self.press = None
        self.prev = None
        self.press_circle = None 
        self.prev_circle = None
        self.composer = composer
        self.x0, self.y0 = self.rect.xy
        self.x0_dot, self.y0_dot = self.circle.get_center()
        self.h = self.rect.get_height()
        self.w = self.rect.get_width()
        self._dpi = composer._dpi 
        self._fig_x , self._fig_y = composer._fig_size

    def __del__(self):
        # self.disconnect()
        pass

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

        self.circlecidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press_circle)
        self.circlecidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release_circle)
        self.circlecidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion_circle)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.rect.contains(event)
        if not contains: return
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        if self.prev is not None:
            xprev, yprev = self.prev
            dx = event.xdata - xprev
            dy = event.ydata - yprev
            if abs(dx + dy) < 0.4/self._fig_x:
                return
        
        
                
                
        self.prev = event.xdata, event.ydata
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress


        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)
        
        self.textbox.set_x(x0+dx+0.5*self.w)
        self.textbox.set_y(y0+dy+0.5*self.h)

        
        self.circle.set_center((self.x0_dot+dx, self.y0_dot))

        self.line.set(xdata=[self.x0+dx+0.5*self.w, self.x0_dot+dx], ydata = [self.y0_dot, self.y0+dy])

        # self.rect.figure.canvas.draw()
        # self.textbox.figure.canvas.draw()
        self.rect.figure.canvas.draw()


    def on_release(self, event):
        'on release we reset the press data'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return

        x1, y1 = event.xdata, event.ydata
        x1, y1 = x1+ 0.5*self.w,y1+ 0.5*self.h

        

        self.rect.set_x(self.x0)
        self.rect.set_y(self.y0)
        
        self.textbox.set_x(self.x0+0.5*self.w)
        self.textbox.set_y(self.y0+0.5*self.h)

        self.circle.set_center((self.x0_dot, self.y0_dot))

        self.line.set(xdata=[self.x0+0.5*self.w, self.x0_dot], ydata = [self.y0_dot, self.y0])

        self.press = None
        self.prev = None

        # self.rect.figure.canvas.draw()
        self.rect.figure.canvas.draw()

        self.composer.updateControlledQubitGate(x1, y1, self)


    def on_press_circle(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.circle.contains(event)
        if not contains: return
        x0, y0 = self.circle.get_center()
        self.press_circle = x0, y0, event.xdata, event.ydata

    def on_motion_circle(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press_circle is None: return
        if event.inaxes != self.circle.axes: return
        if self.prev_circle is not None:
            xprev, yprev = self.prev_circle
            dx = event.xdata - xprev
            dy = event.ydata - yprev
            if abs(dx + dy) < 0.5/self._fig_x:
                return
        
        
                
                
        self.prev_circle = event.xdata, event.ydata
        x0, y0, xpress, ypress = self.press_circle
        dx = event.xdata - xpress
        dy = event.ydata - ypress


        self.rect.set_x(self.x0+dx)
        self.rect.set_y(self.y0)
        
        self.textbox.set_x(self.x0+dx+0.5*self.w)
        self.textbox.set_y(self.y0+0.5*self.h)

        
        self.circle.set_center((self.x0_dot+dx, self.y0_dot+dy))

        self.line.set(xdata=[self.x0+dx+0.5*self.w, self.x0_dot+dx], ydata = [self.y0_dot+dy, self.y0])

        # self.rect.figure.canvas.draw()
        # self.textbox.figure.canvas.draw()
        self.circle.figure.canvas.draw()


    def on_release_circle(self, event):
        'on release we reset the press data'
        if self.press_circle is None: return
        if event.inaxes != self.circle.axes: return

        x1, y1 = event.xdata, event.ydata
        x1, y1 = x1+ 0.5*self.w,y1+ 0.5*self.h

        

        self.rect.set_x(self.x0)
        self.rect.set_y(self.y0)
        
        self.textbox.set_x(self.x0+0.5*self.w)
        self.textbox.set_y(self.y0+0.5*self.h)

        self.circle.set_center((self.x0_dot, self.y0_dot))

        self.line.set(xdata=[self.x0+0.5*self.w, self.x0_dot], ydata = [self.y0_dot, self.y0])

        self.press_circle = None
        self.prev_circle = None

        # self.rect.figure.canvas.draw()
        self.circle.figure.canvas.draw()

        self.composer.updateControlledQubitGateController(x1, y1, self)

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

class DraggableMeasurementTemplate:
    def __init__(self, textbox, rect, arrow, circle, composer):
        self.textbox = textbox
        self.rect = rect
        self.arrow = arrow 
        self.circle = circle
        self.press = None
        self.prev = None
        self.composer = composer
        self.x0, self.y0 = self.rect.xy
        self.x0_dot, self.y0_dot = self.circle.get_center()
        self.h = self.rect.get_height()
        self.w = self.rect.get_width()
        self._dpi = composer._dpi 
        self._fig_x , self._fig_y = composer._fig_size

    def __del__(self):
        # self.disconnect()
        pass

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        # self.cidmotion = self.rect.figure.canvas.mpl_connect(
        #     'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.rect.contains(event)
        if not contains: return
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata


 
            






    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        if self.prev is not None:
            xprev, yprev = self.prev
            dx = event.xdata - xprev
            dy = event.ydata - yprev
            if abs(dx + dy) < 0.5/self._fig_x:
                return
        
        
                
                
        self.prev = event.xdata, event.ydata
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress


        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)
        
        self.textbox.set_x(x0+dx+0.5*self.w - 0.35*self.w)
        self.textbox.set_y(y0+dy+0.5*self.h  - 0.3*self.h)

        
        self.circle.set_center((x0+dx+0.5*self.w, y0+dy+0.7*self.h))

        self.arrow.set(xy=[[x0+dx+0.5*self.w, y0+dy+0.5*self.h], [x0+dx+0.5*self.w+self.w*0.5/2, y0+dy+0.5*self.h-self.h*0.5/2]])

        # self._ax.arrow(pos_x, pos_y+0.2*gate_h, gate_w*0.5/2, -gate_h*0.5/2, zorder=7, width=0.002,head_width=0.004, color="black")
        # self.rect.figure.canvas.draw()
        # self.textbox.figure.canvas.draw()
        self.rect.figure.canvas.draw()


    def on_release(self, event):
        'on release we reset the press data'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return

        x1, y1 = event.xdata, event.ydata
        x1, y1 = x1+ 0.5*self.w,y1+ 0.5*self.h

        x0, y0 = self.x0, self.y0

        self.rect.set_x(x0)
        self.rect.set_y(y0)
        
        self.textbox.set_x(x0+0.5*self.w - 0.35*self.w)
        self.textbox.set_y(y0+0.5*self.h  - 0.3*self.h)

        
        self.circle.set_center((x0+0.5*self.w, y0+0.7*self.h))

        self.arrow.set(xy=[[x0+0.5*self.w, y0+0.5*self.h], [x0+0.5*self.w+self.w*0.5/2, y0+0.5*self.h-self.h*0.5/2]])

        self.press = None
        self.prev = None

        self.rect.figure.canvas.draw()
        # self.rect.figure.canvas.draw()

        self.composer.addMeasurement(x1, y1, self)


class DraggableMeasurement:
    def __init__(self, textbox, rect, arrow, circle, composer, measurement):
        self.textbox = textbox
        self.measurement = measurement
        self.rect = rect
        self.arrow = arrow 
        self.circle = circle
        self.press = None
        self.prev = None
        self.composer = composer
        self.x0, self.y0 = self.rect.xy
        self.x0_dot, self.y0_dot = self.circle.get_center()
        self.h = self.rect.get_height()
        self.w = self.rect.get_width()
        self._dpi = composer._dpi 
        self._fig_x , self._fig_y = composer._fig_size

    def __del__(self):
        # self.disconnect()
        pass

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.rect.contains(event)
        if not contains: return
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata


 
            






    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        if self.prev is not None:
            xprev, yprev = self.prev
            dx = event.xdata - xprev
            dy = event.ydata - yprev
            if abs(dx + dy) < 0.5/self._fig_x:
                return
        
        
                
                
        self.prev = event.xdata, event.ydata
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress


        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)
        
        self.textbox.set_x(x0+dx+0.5*self.w - 0.35*self.w)
        self.textbox.set_y(y0+dy+0.5*self.h  - 0.3*self.h)

        
        self.circle.set_center((x0+dx+0.5*self.w, y0+dy+0.7*self.h))

        self.arrow.set(xy=[[x0+dx+0.5*self.w, y0+dy+0.5*self.h], [x0+dx+0.5*self.w+self.w*0.5/2, y0+dy+0.5*self.h-self.h*0.5/2]])

        # self._ax.arrow(pos_x, pos_y+0.2*gate_h, gate_w*0.5/2, -gate_h*0.5/2, zorder=7, width=0.002,head_width=0.004, color="black")
        # self.rect.figure.canvas.draw()
        # self.textbox.figure.canvas.draw()
        self.rect.figure.canvas.draw()


    def on_release(self, event):
        'on release we reset the press data'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return

        x1, y1 = event.xdata, event.ydata
        x1, y1 = x1+ 0.5*self.w,y1+ 0.5*self.h

        x0, y0 = self.x0, self.y0

        self.rect.set_x(x0)
        self.rect.set_y(y0)
        
        self.textbox.set_x(x0+0.5*self.w - 0.35*self.w)
        self.textbox.set_y(y0+0.5*self.h  - 0.3*self.h)

        
        self.circle.set_center((x0+0.5*self.w, y0+0.7*self.h))

        self.arrow.set(xy=[[x0+0.5*self.w, y0+0.5*self.h], [x0+0.5*self.w+self.w*0.5/2, y0+0.5*self.h-self.h*0.5/2]])

        self.press = None
        self.prev = None

        # self.rect.figure.canvas.draw()
        # self.rect.figure.canvas.draw()

        self.composer.updateMeasurement(x1, y1, self)

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)


GateNameList = ["X",
                "Y",
                "Z",
                "I",
                "S",
                "H",
                "Rx",
                "Ry",
                "Rz",
                "CNOT"]

SingleQubitGates = {
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "I": "I",
    "S": "S",
    "Hadamard": "H",
    "RX": "Rx",
    "RY": "Ry",
    "RZ": "Rz",
}
GateList = {"X":PauliX,
            "Y":PauliY,
            "Z":PauliZ,
            "I":I,
            "S":S,
            "H":Hadamard,
            "Rx":RX,
            "Ry":RY,
            "Rz":RZ,
            "CNOT":CNOT
            }   

ControlledGates = {
    "CRX": "Rx", 
    "CRY": "Ry", 
    "CRZ": "Rz", 
    "CNOT": "X"
}