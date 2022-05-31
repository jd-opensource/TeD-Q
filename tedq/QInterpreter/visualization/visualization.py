import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np

from tedq.QInterpreter.operators.measurement import Expectation

class matplotlib_drawer(object):
    r"""
        This is the visualization module based on the matplotlib package.
    """
    def __init__(self, circuit, **kwargs):


        self._circuit = circuit
        self._fig_size = kwargs.get("figsize", (15,15))
        self._dpi = kwargs.get("dpi", 72)
        self._scale = kwargs.get("scale", 1)
        self._fig_width = self._dpi*self._fig_size[0]
        self._fig_height = self._dpi*self._fig_size[1]
        self._qubit_num = circuit._num_qubits


        # TODO: layer size
        self._plt = plt
        self._fig= plt.figure(figsize=self._fig_size, dpi=self._dpi)
        self._ax = plt.gca()
        self._patch = mpatch


        # set ax environment
        self._ax.axis("off")
        self._ax.set_aspect(1)
        self._ax.tick_params(
            labelbottom=False, labeltop=False, labelleft=False, labelright=False
        )
        self._ax.set_title("")

        layers = self.get_layers(circuit)
        
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

        self._font_size = 10*self._scale*self._dpi/72
        self._line_width = 1.2*self._scale*self._dpi/72



    def _GetXYPosFromXYIndex(self, xi, yi):
        pos_x = xi*self._div_x
        pos_y = yi*self._div_y + 1
        return (pos_x, pos_y)

    def _PlotRectangle(self, x,y,w,h,**kwargs):
        box = self._patch.Rectangle(
            xy=(x-0.5*w, y-0.5*h),
            width=w,
            height=h,
            zorder=3,         #TODO
            **kwargs
        )
        self._ax.add_patch(box)

    def _PlotText(self, x,y,text,**kwargs):
        self._ax.text(
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
    def _PlotParams(self, x,y,param,**kwargs):
        self._ax.text(
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


    def _DrawGate(self,xi, yi, text, params=None, **kwargs):
        (pos_x, pos_y) = self._GetXYPosFromXYIndex(xi, yi)
        gate_w = self._gate_dx
        gate_h = self._gate_dy


        self._PlotRectangle(pos_x, pos_y, gate_w, gate_h, **kwargs)
        if params==None :
            self._PlotText(pos_x, pos_y, text)
        else:
            self._PlotText(pos_x, pos_y-0.1*self._gate_dy, text)
            # self._PlotParams(pos_x, pos_y+0.35*self._gate_dy, params[0])

    def _DrawSingleQubitGate(self,xi, yarr, operator):
        if operator.is_preparation:
            color="darkturquoise"
        else:
            color="skyblue"
        self._DrawGate(xi, yarr[0], SingleQubitGates[operator.name], operator.parameters, color=color)

    def _DrawControlledQubitGate(self,xi, yarr, operator):
        self._DrawGate(xi, yarr[1], ControlledGates[operator.name], color="aquamarine")
        self._DrawConnectionWire(xi, yarr[0], yarr[1])
        self._DrawControlQubit(xi, yarr[0])
    
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

    def _DrawConnectionWire(self,xi, yi, yj):
        (pos_x_i, pos_y_i) = self._GetXYPosFromXYIndex(xi, yi)
        (pos_x_i, pos_y_j) = self._GetXYPosFromXYIndex(xi, yj)
        self._ax.plot(
            [pos_x_i, pos_x_i],
            [pos_y_i, pos_y_j],
            color="black",
            # linewidth=self._wire_width,
            zorder=1,
            linewidth=self._line_width
        ) 

    def     _DrawQuantumWire(self, xi, yi):
        (pos_x, pos_y) = self._GetXYPosFromXYIndex(xi, yi)
        self._ax.plot(
            [pos_x - self._div_x/2, pos_x + self._div_x/2],
            [pos_y, pos_y],
            color="black",
            # linewidth=self._wire_width,
            zorder=1,
            linewidth=self._line_width
        )
    def _DrawMeasurements(self, color="black"):
        measurements = self._circuit.measurements
        if measurements is None:
            return
        head_of_wires = max(self._ParseCircuitHead())
        
        for measurement in measurements:
            if measurement.return_type == Expectation:
                (pos_x, pos_y) = self._GetXYPosFromXYIndex(head_of_wires, measurement.obs.qubits[0])
                self._DrawQuantumWire(head_of_wires-0.5, measurement.obs.qubits[0])
                gate_w = self._gate_dx
                gate_h = self._gate_dy  
    

                self._PlotRectangle(pos_x, pos_y, gate_w, gate_h, facecolor="white", edgecolor="black",linewidth = self._line_width*0.5)    

                circle = self._patch.Arc(
                    xy=(pos_x, pos_y+0.2*gate_h),
                    width=-gate_w*0.8,
                    height=-gate_h*0.8,
                    # fc=fc,
                    # ec=ec,
                    linewidth=self._line_width*0.5,
                    zorder=6,
                    theta1=0,
                    theta2=180,
                    color="black",
                )
                
                self._ax.text(
                    pos_x - 0.35 * gate_w,
                    pos_y - 0.3 * gate_h,
                    SingleQubitGates[measurement.obs.name],
                    ha="center",
                    va="center",
                    fontsize=self._font_size,
                    # color=gt,
                    clip_on=True,
                    zorder=7,
                )
                self._ax.add_patch(circle)
                self._ax.arrow(pos_x, pos_y+0.2*gate_h, gate_w*0.5/2, -gate_h*0.5/2, zorder=7, width=abs(gate_w*0.2*0.01),head_width=abs(gate_w*0.8*0.01), color="black")
    def _ParseCircuitHead(self):
        operators = self._circuit.operators
        head_of_wires = [0 for i in range(self._qubit_num)]
        for op in operators:
            covered_wires = self._CoveredWires(op)
            cur_head = max([head_of_wires[i] for i in covered_wires])
            while cur_head<op._expected_index:
                cur_head+=1
                
                    
            for wire in covered_wires:
                head_of_wires[wire] = cur_head+1
        self.max_qubit_idx = head_of_wires
        return head_of_wires


    def draw_circuit(self):
        layers = self.parse_circuit(self._circuit)
        for i, layer in enumerate(layers):
            for j in range(self._qubit_num):
                self._DrawQuantumWire(i, j)
                
            for op in layer:
                if (op.name in SingleQubitGates):
                    self._DrawSingleQubitGate(i, op.qubits, op)
                elif (op.name in ControlledGates):
                    # print(op.qubits)
                    self._DrawControlledQubitGate(i, op.qubits, op)
        self._DrawMeasurements()


    def full_draw(self):
        r"""
        Draw full circuit including initial states, gates, and measurements.
        """
        # gate_layers, measurement_layers = self.get_layers(self._circuit)
        gate_layers = self.parse_circuit(self._circuit)
        measurements = self.get_measurement(self._circuit)
        layer_width = self._layer_width
        qubit_length = 0
        x_pos_count = self._layer_width
        y_count = 0
        self.draw_wire_number(layer_width/2)

        for layer in gate_layers:
            # get layer width
            
            x_pos = x_pos_count + layer_width / 2
            self.draw_qubits(x_pos_count, layer_width)
            for op in layer:
                # draw each op
                self.draw_op(op, x_pos)

            x_pos_count += layer_width
        x_pos = x_pos_count + layer_width / 2
        self.draw_qubits(x_pos_count, layer_width/2)
        self.draw_measurement(x_pos, measurements)

    def draw(self):
        r"""
        afsd
        """
        # x_axis is the horizontal axis pointing to right -> and y_axis is the vertical axis pointing to upward ↑.
        layers = self.get_layers(self._circuit)
        qubit_length = 0
        x_count = 0
        y_count = 0
        for layer in layers:
            # get layer width
            for op in layer:
                # draw each op
                pass
        layer_width = 0.6

        self.draw_qubits(0, layer_width)
        self.draw_connection(layer_width / 2, [1, 2], color="skyblue")
        self.draw_control_qubit(layer_width / 2, 1)
        self.draw_single_qubit_op(layer_width / 2, 3)
        self.draw_qubits(layer_width, layer_width)
        self.draw_connection(3 * layer_width / 2, [0, 3], color="skyblue")
        self.draw_control_qubit(3 * layer_width / 2, 3)
        self.draw_single_qubit_op(3 * layer_width / 2, 0)

        return self._fig
    @classmethod
    def parse_circuit(cls, circuit):
        head_of_wires = [0 for i in range(circuit._num_qubits)]
        wires = []
        for op in circuit.operators:
            covered_wires = cls._covered_wires(op)
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

    @classmethod
    def _covered_wires(cls, operator):
        qubit_min = min(operator.qubits)
        qubit_max = max(operator.qubits)
        return [i for i in range(qubit_min, qubit_max + 1)]

    @classmethod
    def get_layers(cls, circuit):
        layer_count = [0 for i in range(circuit._num_qubits)]
        layers = []
        for op in circuit.operators:
            all_qubits = cls._covered_qubits(op.qubits)
            index = max([layer_count[i] for i in all_qubits]) + 1
            for qubit in all_qubits:
                layer_count[qubit] = index
            if len(layers) < index:
                layers.append([op])
            else:
                layers[index - 1].append(op)
            # print(layers)
        return layers
    @classmethod
    def _CoveredWires(cls, operator):
        qubit_min = min(operator.qubits)
        qubit_max = max(operator.qubits)
        return [i for i in range(qubit_min, qubit_max + 1)]

    @classmethod
    def get_measurement(cls, circuit):
        layer_count = [0 for i in range(circuit._num_qubits)]
        layers = []
        for op in circuit.measurements:
            all_qubits = cls._covered_qubits(op.obs.qubits)
            index = max([layer_count[i] for i in all_qubits]) + 1
            for qubit in all_qubits:
                layer_count[qubit] = index
            if len(layers) < index:
                layers.append([op.obs])
            else:
                layers[index - 1].append(op.obs)
        if len(layers)>0:
            return layers[0]
        else:
            return None

    @staticmethod
    def _covered_qubits(qubits):
        qubit_min = min(qubits)
        qubit_max = max(qubits)
        return [i for i in range(qubit_min, qubit_max + 1)]
    
    def draw_wire_number(self, stat_x_pos ):
        y0s = [i * self._line_inter_val for i in range(self._qubit_num)]
        x0 = stat_x_pos
        i = len(y0s)
        for y0 in y0s:
            
            self._ax.text(
                x0,
                y0,
                "  "+str(i)+"   |0>",
                ha="center",
                va="center",
                fontsize=self._font_size,
                # color=gt,
                clip_on=False,
                zorder=7,
            )
            i-=1
    def draw_measurement(self, stat_x_pos, measurements, color="black"):
        if measurements is None:
            return
        y0s = [i * self._line_inter_val for i in range(self._qubit_num)]
        y0s.reverse()
        x0 = stat_x_pos
        wid = self._basic_gate_width
        hig = wid
        i=0
        for y0 in y0s:
            circle = self._patch.Arc(
                xy=(x0, y0-hig*0.9/4),
                width=wid*0.8,
                height=hig*0.8,
                # fc=fc,
                # ec=ec,
                # linewidth=self._lwidth15,
                zorder=6,
                theta1=0,
                theta2=180,
                color=color,
            )
            box = self._patch.Rectangle(
                xy=(x0 - 0.5 * wid, y0 - 0.5 * hig),
                width=wid,
                height=hig,
                # fc=fc,
                # ec=ec,
                # linewidth=self._lwidth15,
                zorder=5,
                edgecolor=color,
                facecolor="white",
            )
            if i < len(measurements): # Not every line has a measurement
                self._ax.text(
                    x0 - 0.35 * wid,
                    y0 + 0.3 * hig,
                    SingleQubitGates[measurements[i].name],
                    ha="center",
                    va="center",
                    fontsize=self._font_size,
                    # color=gt,
                    clip_on=True,
                    zorder=7,
                )
            self._ax.add_patch(circle)
            self._ax.add_patch(box)
            self._ax.arrow(x0, y0-hig*0.9/4, wid*0.5/2, hig*0.8/2, zorder=7, width=0.01,head_width=0.05, color=color)
            i+=1
    def draw_qubits(self, stat_x_pos, width):
        y0s = [i * self._line_inter_val for i in range(self._qubit_num)]
        x0 = stat_x_pos
        for y0 in y0s:
            
            self._ax.plot(
                [x0, x0 + width],
                [y0, y0],
                color="black",
                linewidth=self._line_width,
                zorder=1,
            )

    def draw_op(self, op, xpos):
        if (op.name in SingleQubitGates):
            self.draw_single_qubit_op_with_params(xpos, op)
        elif (op.name in ControlledGates):
            self.draw_control_op(xpos, op.qubits, ControlledGates[op.name], color="lightcoral")
        pass

    def draw_control_op(self, xpos, qubits, text, color="skyblue"):
        self.draw_connection(xpos, qubits, color=color)
        self.draw_control_qubit(xpos, qubits[1], color=color)
        self.draw_single_qubit_op(xpos, qubits[0], text=text, color=color)
        pass

    def draw_swap(self):
        pass

    def draw_control_qubit(self, xpos, qubit, color="skyblue"):
        xpos = xpos
        if not self._reverse_qubit:
            qubit = self._qubit_num - 1 - qubit
        ypos = qubit * self._line_inter_val
        box = self._patch.Circle(
            xy=(xpos, ypos), radius=self._control_radius, color=color, zorder=3
        )
        self._ax.add_patch(box)

    def draw_connection(self, xpos, qubits, color="skyblue"):
        if not self._reverse_qubit:
            qubits = [self._qubit_num - 1 - q for q in qubits]
        y0 = min(qubits) * self._line_inter_val
        y1 = max(qubits) * self._line_inter_val
        self._ax.plot(
            [xpos, xpos], [y0, y1], linewidth=self._line_width, color=color, zorder=2
        )

    def draw_single_qubit_op(self, xpos, qubit, text=None, color="skyblue"):
        xpos = xpos
        if not self._reverse_qubit:
            qubit = self._qubit_num - 1 - qubit
        ypos = qubit * self._line_inter_val
        wid = self._basic_gate_width
        hig = wid
        box = self._patch.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - 0.5 * hig),
            width=wid,
            height=hig,
            # fc=fc,
            # ec=ec,
            # linewidth=self._lwidth15,
            zorder=3,
            color=color,
        )
        self._ax.text(
            xpos,
            ypos,
            text,
            ha="center",
            va="center",
            fontsize=self._font_size,
            # color=gt,
            clip_on=True,
            zorder=4,
        )
        self._ax.add_patch(box)

    def draw_single_qubit_op_with_params(self, xpos, operator, color="skyblue"):
        xpos = xpos
        if operator.is_preparation:
            color="darkturquoise"
        qubit = operator.qubits[0]
        if not self._reverse_qubit:
            qubit = self._qubit_num - 1 - qubit
        ypos = qubit * self._line_inter_val
        wid = self._basic_gate_width
        hig = wid
        box = self._patch.Rectangle(
            xy=(xpos - 0.5 * wid, ypos - 0.5 * hig),
            width=wid,
            height=hig,
            # fc=fc,
            # ec=ec,
            # linewidth=self._lwidth15,
            zorder=3,
            color=color,
        )
        gateText = self._ax.text(
            xpos,
            ypos,
            SingleQubitGates[operator.name],
            ha="center",
            va="center",
            fontsize=14,
            # color=gt,
            clip_on=True,
            zorder=4,
        )
        if len(operator.parameters)>0:
            gateText.set_y(ypos+0.15*hig)
            self._ax.text(
                xpos,
                ypos-0.35*hig,
                "("+str(np.round(float(operator.parameters[0]),3))+")",
                ha="center",
                va="center",
                fontsize=8,
                # color=gt,
                clip_on=True,
                zorder=5,
            )
        self._ax.add_patch(box)


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

ControlledGates = {
    "CRX": "Rx", 
    "CRY": "Ry", 
    "CRZ": "Rz", 
    "CNOT": "X"
}
