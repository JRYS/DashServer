#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
This example demonstrates a simple web server providing visualization of data
from a USB-201 device for a single client.  It makes use of the Dash
Python framework for web-based interfaces and a plotly graph.  To install the
dependencies for this example, run:
   $ pip install dash

Running this example:
1. Start the server by running the web_server.py module in a terminal. For example,
   $ ./web_server.py
2. Open a web browser on a device on the same network as the host device and
   enter http://<host>:8080 in the address bar,
   replacing <host> with the IP Address or hostname of the host device.

Stopping this example:
1. To stop the server, use Ctrl+C in the terminal window where the server
   was started.
"""

import json
import plotly.graph_objs as go
import numpy as np
from collections import deque
from dash import Dash, dcc, html, ctx
from dash.dependencies import Input, Output, State

app = Dash(__name__)
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

def init_chart_data(number_of_channels, number_of_samples, sample_rate):
    """
    Initializes the chart with the specified number of samples.

    Args:
        number_of_channels (int): The number of channels to be displayed.
        number_of_samples (int): The number of samples to be displayed.
        sample_rate (float): The current sampling rate

    Returns:
        str: A string representation of a JSON object containing the chart data.
    """
    t = 1 / sample_rate
    samples = []
    for i in range(number_of_samples):
        samples.append(i * t)
    data = []
    for _ in range(number_of_channels):
        data.append([None] * number_of_samples)

    chart_data = {'data': data, 'samples': samples, 'sample_count': 0}

    return json.dumps(chart_data)


###############################################################################
# Define the HTML layout for the user interface, consisting of
# dash-html-components and dash-core-components.
# pylint: disable=no-member
app.layout = html.Div([
    html.H1(
        children='Dash Web Server Example',
        id='exampleTitle'
    ),
    html.Div([
        html.Div(
            id='rightContent',
            children=[
                dcc.Graph(id='stripChart', style={'height': 600}),
                html.Div(id='errorDisplay',
                         children='',
                         style={'font-weight': 'bold', 'color': 'red'}),
            ], style={'width': '100%', 'box-sizing': 'border-box',
                      'float': 'left', 'padding-left': 320}
        ),
        html.Div(
            id='leftContent',
            children=[
                html.Label('Sample Rate (Hz)',
                           style={'font-weight': 'bold', 'display': 'block',
                                  'margin-top': 10}),
                dcc.Input(id='sampleRate', type='number', step=1, value=1000.0,
                          style={'width': 100, 'display': 'block'}),
                html.Label('Number of Samples',
                           style={'font-weight': 'bold',
                                  'display': 'block', 'margin-top': 10}),
                dcc.Input(id='numberOfSamples', type='number', step=1, value=1000,
                          style={'width': 100, 'display': 'block'}),
                html.Br(),
                html.P(id='textOut'),
                html.Label('Active Channels',
                           style={'font-weight': 'bold', 'display': 'block',
                                  'margin-top': 10}),
                dcc.Checklist(
                    id='channelSelections',
                    options=[
                        {'label': 'Waveform 0', 'value': 0},
                        {'label': 'Waveform 1', 'value': 1},
                        {'label': 'Waveform 2', 'value': 2},
                        {'label': 'Waveform 3', 'value': 3},
                        {'label': 'Waveform 4', 'value': 4},
                        {'label': 'Waveform 5', 'value': 5},
                        {'label': 'Waveform 6', 'value': 6},
                        {'label': 'Waveform 7', 'value': 7},
                    ],
                    labelStyle={'display': 'block'},
                    value=[0]
                ),
                html.H4(children='Create complex waveform.'),
                dcc.Checklist(
                    id='sumChannels',
                    options=[{'label': 'Complex waveform', 'value': 0}, ],
                    labelStyle={'display': 'block'},
                    value=[-1]
                ),
                html.Div([
                    html.Button(
                        children='Start',
                        id='startButton',
                        style={'width': 100, 'height': 25, 'text-align': 'center',
                               'margin-top': 10}
                    ),
                    html.Br(),
                    html.Button(
                        children='Stop',
                        id='stopButton',
                        style={'width': 100, 'height': 25, 'text-align': 'center',
                               'margin-top': 10}
                    ),
                    html.Div(id='button_container')
                ]),
            ], style={'width': 320, 'box-sizing': 'border-box', 'padding': 10,
                      'position': 'absolute', 'top': 0, 'left': 0}
        ),
    ], style={'position': 'relative', 'display': 'block',
              'overflow': 'hidden'}),
    dcc.Interval(
        id='timer',
        interval=1000 * 60 * 60 * 24,  # in milliseconds
        n_intervals=0
    ),
    html.Div(
        id='chartData',
        style={'display': 'none'},
        children=init_chart_data(1, 1000, 1000)
    ),
    html.Div(
        id='chartInfo',
        style={'display': 'none'},
        children=json.dumps({'sample_count': 1000})
    ),
    html.Div(
        id='status',
        style={'display': 'none'}
    ),
])


# pylint: enable=no-member
@app.callback(
    Output('status', 'children'),
    [Input('startButton', 'n_clicks'),
     Input('stopButton', 'n_clicks')],
    [State('startButton', 'children'),
     State('channelSelections', 'value'),
     State('sampleRate', 'value'),
     State('numberOfSamples', 'value')]
)
def start_stop_click(btn1, btn2, btn1_label, active_channels,
                     sample_rate, number_of_samples):
    """
     A callback function to change the application status when the Ready,
     Start or Stop button is clicked.

     Args:
         btn1 (int): button id
         btn2 (int): button id
         btn1_label (str): The current label on the button.
         active_channels ([int]): A list of integers corresponding to the user
             selected Active channel checkboxes.
         sample_rate (float): current sample rate
         number_of_samples (int):

     Returns:
         str: The new application status - "idle", "configured", "running"
         or "error"

     """
    button_clicked = ctx.triggered_id
    output = 'idle'
    if btn1 is not None and btn1 > 0:
        if button_clicked == 'startButton' and btn1_label == 'Configure':
            if number_of_samples is not None and sample_rate is not None and active_channels:
                if (999 < number_of_samples <= 10000) and (999 < sample_rate <= 10000):
                    output = 'configured'
                else:
                    output = 'error'
            else:
                output = 'error'
        elif button_clicked == 'startButton' and btn1_label == 'Start':
            output = 'running'

    if btn2 is not None and btn2 > 0:
        if button_clicked == 'stopButton':
            output = 'idle'

    return output


@app.callback(
    Output('timer', 'interval'),
    [Input('status', 'children'),
     Input('chartData', 'children'),
     Input('chartInfo', 'children')],
    [State('sampleRate', 'value'),
     State('numberOfSamples', 'value'), ]
)
def update_timer_interval(acq_state, chart_data_json_str, chart_info_json_str,
                          sample_rate, number_of_samples):
    """
    A callback function to update the timer interval.  The timer is temporarily
    disabled while processing data by setting the interval to 1 day and then
    re-enabled when the data read has been plotted.  The interval value when
    enabled is calculated based on the sample rate and some trial and error

    Args:
        acq_state (str): The application state of "idle", "configured",
            "running" or "error" - triggers the callback.
        chart_data_json_str (str): A string representation of a JSON object
            containing the current chart data - triggers the callback.
        chart_info_json_str (str): A string representation of a JSON object
            containing the current chart status - triggers the callback.
        sample_rate (float): Waveform sample rate
        number_of_samples (float): The number of samples per waveform cycle.

    Returns:
        refresh_rate (int): timer tick interval
    """

    chart_data = json.loads(chart_data_json_str)
    chart_info = json.loads(chart_info_json_str)

    refresh_rate = 1000 * 60 * 60 * 24  # 1 day

    if acq_state == 'running':
        # Activate the timer when the sample count displayed to the chart
        # matches the sample count of any new data.

        if 0 < chart_info['sample_count'] == chart_data['sample_count']:

            refresh_rate = int(number_of_samples / sample_rate * 1000)

            if refresh_rate < 10:
                refresh_rate = 10

    return refresh_rate


@app.callback(
    Output('sampleRate', 'disabled'),
    [Input('status', 'children')]
)
def disable_sample_rate_input(acq_state):
    """
    A callback function to disable the sample rate input when the
    application status changes to configured or running.
    """
    disabled = False
    if acq_state == 'configured' or acq_state == 'running':
        disabled = True
    return disabled


@app.callback(
    Output('numberOfSamples', 'disabled'),
    [Input('status', 'children')]
)
def disable_samples_to_disp_input(acq_state):
    """
    A callback function to disable the number of samples to display input
    when the application status changes to configured or running.
    """
    disabled = False
    if acq_state == 'configured' or acq_state == 'running':
        disabled = True
    return disabled


@app.callback(
    Output('channelSelections', 'options'),
    [Input('status', 'children')]
)
def disable_channel_checkboxes(acq_state):
    """
    A callback function to disable the active channel checkboxes when the
    application status changes to configured or running.
    """
    num_of_checkboxes = 8
    frequencies = [1.0, 2.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]  # Hz
    options = []
    for channel in range(num_of_checkboxes):
        label = str(frequencies[channel]) + ' Hz sine wave '
        disabled = False
        if acq_state == 'configured' or acq_state == 'running':
            disabled = True
        options.append({'label': label, 'value': channel, 'disabled': disabled})
    return options


@app.callback(
    Output('sumChannels', 'options'),
    Input('status', 'children')
)
def disable_sum_channels(acq_state):
    """
    A callback function to disable the make_complex checkbox when the
    application status changes to configured or running.
    """
    num_of_checkboxes = 1
    options = []
    for channel in range(num_of_checkboxes):
        label = 'Complex waveform'
        disabled = False
        if acq_state == 'configured' or acq_state == 'running':
            disabled = True
        options.append({'label': label, 'value': channel, 'disabled': disabled})
    return options

@app.callback(
    Output('startButton', 'disabled'),
    Input('status', 'children')
)
def disable_configure_start(acq_state):
    """
    A callback function to disable the start button when the
    application status changes to running.
    """
    disabled = False
    if acq_state == 'running':
        disabled = True
    return disabled

@app.callback(
    Output('startButton', 'children'),
    [Input('status', 'children')]
)
def update_start_stop_button_name(acq_state):
    """
    A callback function to update the label on the button when the application
    status changes.

    Args:
        acq_state (str): The application state of "idle", "configured",
            "running" or "error" - triggers the callback.

    Returns:
        str: The new button label of "Ready", "Start" or "Stop"
    """

    output = 'Configure'
    if acq_state == 'configured' or acq_state == 'running':
        output = 'Start'
    elif acq_state == 'idle':
        output = 'Configure'
    return output

@app.callback(
    Output('chartData', 'children'),
    [Input('timer', 'n_intervals'),
     Input('status', 'children')],
    [State('chartData', 'children'),
     State('numberOfSamples', 'value'),
     State('sampleRate', 'value'),
     State('channelSelections', 'value'),
     State('sumChannels', 'value')],
    #prevent_initial_call=True
)
def update_strip_chart_data(_n_intervals, acq_state, chart_data_json_str,
                            number_of_samples, sample_rate, active_channels, make_complex):
    """
    A callback function to update the chart data stored in the chartData HTML
    div element.  The chartData element stores the existing data
    values, allowing data to be shared between callback functions. Global
    variables cannot be used to share data between callbacks (see
    https://dash.plotly.com/basic-callbacks).

    Args:
        _n_intervals (int): Number of timer intervals - triggers the callback.
        acq_state (str): The application state of "idle", "configured",
            "running" or "error" - triggers the callback.
        chart_data_json_str (str): A string representation of a JSON object
            containing the current chart data.
        number_of_samples (float): The number of samples to be displayed.
        active_channels ([int]): A list of integers corresponding to the user
            selected active channel numbers.
        sample_rate (float): the current sample rate
        make_complex ([int]): A list of one checkbox that when selected,
            generates a complex waveform

    Returns:
        str: A string representation of a JSON object containing the updated
        chart data.
    """

    updated_chart_data = chart_data_json_str
    samples_to_display = int(number_of_samples)
    num_channels = len(active_channels)

    if acq_state == 'running':
        chart_data = json.loads(chart_data_json_str)

        frequencies = [1.0, 2.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]  # Hz

        duration = float(number_of_samples / sample_rate)
        start_time = duration * _n_intervals

        t = np.linspace(start_time, start_time + duration, int(sample_rate * duration), endpoint=False)

        sine_waves = np.array([np.sin(2 * np.pi * f * t) for f in frequencies])

        data = sine_waves[active_channels]

        if len(make_complex) > 1:
            temp = sum(data)
            num_channels = 1
            sample_count = add_samples_to_data(samples_to_display, num_channels,
                                               chart_data, temp.flatten(), sample_rate)
        else:

            sample_count = add_samples_to_data(samples_to_display, num_channels,
                                               chart_data, data.T.flatten(), sample_rate)

        # Update the total sample count.
        chart_data['sample_count'] = sample_count

        updated_chart_data = json.dumps(chart_data)

    elif acq_state == 'configured':
        # Clear the data in the strip chart when Ready is clicked.
        updated_chart_data = init_chart_data(num_channels, int(number_of_samples), sample_rate)

    return updated_chart_data


def add_samples_to_data(number_of_samples, num_chans, chart_data, data, sample_rate):
    """
    Adds the samples read from the simulated device to the chart_data object
    used to update the strip chart.

    Args:
        number_of_samples (int): The number of samples to be displayed.
        num_chans (int): The number of selected channels.
        chart_data (dict): A dictionary containing the data that updates the
            strip chart display.
        data: A list to hold available device data
        sample_rate (float): the current sample rate

    Returns:
        int: The updated total sample count after the data is added.

    """

    num_samples_read = int(len(data) / num_chans)
    current_sample_count = int(chart_data['sample_count'])

    if num_samples_read < num_chans:
        return current_sample_count

    # Convert lists to deque objects with the maximum length set to the number
    # of samples to be displayed.  This will pop off the oldest data
    # automatically when new data is appended.
    chart_data['samples'] = deque(chart_data['samples'],
                                  maxlen=number_of_samples)
    for chan in range(num_chans):
        chart_data['data'][chan] = deque(chart_data['data'][chan],
                                         maxlen=number_of_samples)

    start_sample = 0
    if num_samples_read > number_of_samples:
        start_sample = num_samples_read - number_of_samples

    for sample in range(start_sample, num_samples_read):
        chart_data['samples'].append(float(current_sample_count + sample) * 1 / sample_rate)

        for chan in range(num_chans):
            data_index = sample * num_chans + chan
            chart_data['data'][chan].append(data[data_index])

    # Convert deque objects back to lists to be written to div
    # element.
    chart_data['samples'] = list(chart_data['samples'])
    for chan in range(num_chans):
        chart_data['data'][chan] = list(chart_data['data'][chan])

    return current_sample_count + num_samples_read


@app.callback(
    Output('stripChart', 'figure'),
    [Input('chartData', 'children')],
    [State('channelSelections', 'value')]
)
def update_strip_chart(chart_data_json_str, active_channels):
    """
    A callback function to update the strip chart display when new data is read.

    Args:
        chart_data_json_str (str): A string representation of a JSON object
            containing the current chart data - triggers the callback.
        active_channels ([int]): A list of integers corresponding to the user
            selected Active channel checkboxes.

    Returns:
        object: A figure object for a dash-core-components Graph, updated with
        the most recently read data.
    """
    data = []
    xaxis_range = [0, 1000]
    chart_data = json.loads(chart_data_json_str)
    if 'samples' in chart_data and chart_data['samples']:
        xaxis_range = [min(chart_data['samples']), max(chart_data['samples'])]

    if 'data' in chart_data:
        data = chart_data['data']

    plot_data = []
    colors = ['#DD3222', '#FFC000', '#3482CB', '#FF6A00',
              '#75B54A', '#808080', '#6E1911', '#806000']
    # Update the serie data for each active channel.
    for chan_idx, channel in enumerate(active_channels):
        scatter_serie = go.Scatter(
            x=list(chart_data['samples']),
            y=list(data[chan_idx]),
            name='Waveform {0:d}'.format(channel),
            marker={'color': colors[channel]}
        )
        plot_data.append(scatter_serie)

    figure = {
        'data': plot_data,
        'layout': go.Layout(
            xaxis=dict(title='Time (s)', range=xaxis_range),
            yaxis=dict(title='Voltage (V)'),
            margin={'l': 40, 'r': 40, 't': 50, 'b': 40, 'pad': 0},
            showlegend=True,
            title='Strip Chart'
        )
    }

    return figure


@app.callback(
    Output('chartInfo', 'children'),
    Input('stripChart', 'figure'),
    [State('chartData', 'children')]
)
def update_chart_info(_figure, chart_data_json_str):
    """
    A callback function to set the sample count for the number of samples that
    have been displayed on the chart.

    Args:
        _figure (object): A figure object for a dash-core-components Graph for
            the strip chart - triggers the callback.
        chart_data_json_str (str): A string representation of a JSON object
            containing the current chart data - triggers the callback.

    Returns:
        str: A string representation of a JSON object containing the chart info
        with the updated sample count.

    """

    chart_data = json.loads(chart_data_json_str)
    chart_info = {'sample_count': chart_data['sample_count']}

    return json.dumps(chart_info)


@app.callback(
    Output('errorDisplay', 'children'),
    [Input('status', 'children')],
    [State('sampleRate', 'value'),
     State('numberOfSamples', 'value'),
     State('channelSelections', 'value')]
)  # pylint: disable=too-many-arguments
def update_error_message(acq_state, sample_rate, number_of_samples, active_channels):
    """
    A callback function to display error messages.

    Args:
        acq_state (str): The application state of "idle", "configured",
            "running" or "error" - triggers the callback.
        sample_rate (float): The user specified sample rate value.
        number_of_samples (float): The number of samples to be displayed.
        active_channels ([int]): A list of integers corresponding to the user
            selected Active channel checkboxes.

    Returns:
        str: The error message to display.

    """
    error_message = ''
    if acq_state == 'error':
        num_active_channels = len(active_channels)
        _max = 10000
        _min = 1000
        if num_active_channels <= 0:
            error_message += 'Invalid channel selection (min 1); '

        if sample_rate > _max:
            error_message += 'Invalid Sample Rate (max: '
            error_message += str(_max) + '); '
        if sample_rate < _min:
            error_message += 'Invalid Sample Rate (min: '
            error_message += str(_min) + '); '

        if number_of_samples > _max:
            error_message += 'Invalid Samples to display (range: 1000-10000); '
        if number_of_samples < _min:
            error_message += 'Invalid Samples to display (range: 1000-10000); '
    return error_message


@app.callback(
    Output('textOut', 'children'),
    [Input('status', 'children')],
    [State('sampleRate', 'value'),
     State('numberOfSamples', 'value'), ]
)  # pylint: disable=too-many-arguments
def update_text(acq_state, sample_rate, number_of_samples):
    """
    A callback function to display X-Axis time span.

    Args:
        acq_state (str): The application state of "idle", "configured",
            "running" or "error" - triggers the callback.
        sample_rate (float): The user specified sample rate value.
        number_of_samples (float): The number of samples to be displayed.

    Returns:
        str: X-axis time span.

    """

    t = 1 / sample_rate
    span = number_of_samples * t

    return f'Time Span: {span:.4f} sec.'


if __name__ == '__main__':
    app.run(debug=False, use_reloader=True)
