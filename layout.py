import PySimpleGUI as sg

config_names_layout = [
    [sg.Text("Network type:")],
    [sg.Text("Optimizer:")],
    [sg.Text("Hidden units:")],
    [sg.Text("Weight decay:")],
    [sg.Text("Dropout:")],
    [sg.Text("Batch size:")],
    [sg.Text("Learning rate:")],
    [sg.Text("Device:")]
]

config_elements_layout = [
    [
        sg.Radio("Shallow MLP", "net_type", True),
        sg.Radio("Deep MLP", "net_type"),
        sg.Radio("Shallow CNN", "net_type"),
        sg.Radio("Deep CNN", "net_type")
    ],
    [
        sg.Radio("Stochastic Gradient Descent (SGD)", "optimizer", True),
        sg.Radio("Adam", "optimizer")
    ],
    [sg.Input("1000")],
    [sg.Input("0.0")],
    [sg.Input("0.0")],
    [sg.Input("128")],
    [sg.Input("0.001")],
    [
        sg.Radio("CPU", "device", True),
        sg.Radio("GPU 0", "device"),
        sg.Radio("GPU 1", "device")
    ],
    [sg.Checkbox("Save plot", default=False)],
    [sg.Checkbox("Save model", default=False)],
    [sg.Button("START", key="START")]
]

config_layout = [
    [
        sg.Column(config_names_layout, vertical_alignment="t"),
        sg.Column(config_elements_layout, vertical_alignment="t")
    ]
]

plot_layout = [
    [
        sg.Column([
            [sg.Canvas(key="-CANVAS1-")],
            [sg.Canvas(key="-CANVAS2-")]
        ], size=(180, 30), key="Column", expand_x=True, expand_y=True, scrollable=True,  vertical_scroll_only=True)
    ]
]

terminal_layout = [
    [sg.Output(size=(80, 24))]
]

layout = [
    [
        sg.Column(config_layout),
        sg.VSeperator(),
        sg.Column(terminal_layout)
    ],
    [
        sg.HSeparator()
    ],
    plot_layout
]