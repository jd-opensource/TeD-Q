import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np

class new_matplotlib_drawer(object):
    r"""
        This is the visualization module based on the matplotlib package.
    """
    def __init__(self, circuit, **kwargs):
        self._circuit = circuit
        self._qubit_num = circuit._num_qubits
        layers = self.get_layers(circuit)
        self._scale = 1
        self._plt = plt
        self._patch = mpatch
        self._line_width = 2*self._scale
        self._layer_width = 0.6*self._scale
        self._line_inter_val = 0.5*self._scale
        self._basic_gate_width = 0.4*self._scale
        self._control_radius = 0.05*self._scale
        self._layer_blank_width = 0.05*self._scale
        self._reverse_qubit = False
        figWidth = np.shape(layers)[0]
        self._fig = plt.figure(figsize=(figWidth+2,8))
        self._ax = self._fig.add_subplot(111)
        self._ax.axis("off")
        self._ax.set_aspect("equal")
        self._ax.tick_params(
            labelbottom=False, labeltop=False, labelleft=False, labelright=False
        )

    def full_draw(self):
        r"""
        Draw full circuit including initial states, gates, and measurements.
        """
        # gate_layers, measurement_layers = self.get_layers(self._circuit)
        gate_layers = self.get_layers(self._circuit)
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
            # print(layers)
        return layers[0]

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
                fontsize=10,
                # color=gt,
                clip_on=False,
                zorder=7,
            )
            i-=1
    def draw_measurement(self, stat_x_pos, measurements, color="black"):
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
            self._ax.text(
                x0 - 0.35 * wid,
                y0 + 0.3 * hig,
                SingleQubitGates[measurements[i].name],
                ha="center",
                va="center",
                fontsize=10,
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
        self.draw_control_qubit(xpos, qubits[0], color=color)
        self.draw_single_qubit_op(xpos, qubits[1], text=text, color=color)
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
            fontsize=14,
            # color=gt,
            clip_on=True,
            zorder=4,
        )
        self._ax.add_patch(box)
        pass

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
        pass
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

ControlledGates = {"CRX": "Rx", "CRY": "Ry", "CRZ": "Rz", "CNOT": "X"}
