import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "MELDNa Prediction"

# Define the CSS styles
styles = {
    'container': {
        'width': '50%',
        'margin': '0 auto',
        'textAlign': 'center',
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': '#f9f9f9',
        'border': '1px solid #ddd',
        'borderRadius': '8px',
        'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
    },
    'header': {
        'fontSize': '2.5em',
        'marginBottom': '20px',
        'color': '#333',
        'fontWeight': 'bold'
    },
    'label': {
        'fontSize': '1.2em',
        'marginBottom': '5px',
        'display': 'block',
        'color': '#555'
    },
    'input': {
        'width': '100%',
        'padding': '10px',
        'marginBottom': '20px',
        'fontSize': '1em',
        'border': '1px solid #ccc',
        'borderRadius': '4px',
        'boxShadow': 'inset 0 1px 3px rgba(0, 0, 0, 0.1)'
    },
    'button': {
        'backgroundColor': '#4CAF50',
        'color': 'white',
        'padding': '15px 20px',
        'fontSize': '1em',
        'border': 'none',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'transition': 'background-color 0.3s'
    },
    'button:hover': {
        'backgroundColor': '#45a049'
    },
    'output': {
        'marginTop': '20px',
        'fontSize': '1.2em',
        'color': '#333',
        'backgroundColor': '#fff',
        'padding': '10px',
        'border': '1px solid #ddd',
        'borderRadius': '4px',
        'boxShadow': '0 1px 2px rgba(0, 0, 0, 0.1)'
    },
    'info': {
        'fontSize': '0.9em',
        'color': '#777',
        'marginBottom': '20px'
    }
}

app.layout = html.Div(style=styles['container'], children=[
    html.H1("MELDNa Prediction", style=styles['header']),

    html.Div([
        html.Label("MELDNa Score", style=styles['label']),
        html.Div("The scores should be in the format [a1, a2, .. , aN]. "
                 "This is the MELD of a patient from day 1 to day N. Currently, we support N being [1, 3, 5, 7].",
                 style=styles['info']),
        dcc.Input(id='meldna-score', type='text', style=styles['input'], value=''),
    ]),

    html.Div([
        html.Label("Time Stamps", style=styles['label']),
        html.Div("The timestamps should be in the format [a1, a2, .. , aN]. "
                 "This is the timestamps of the MELD measurement of a patient from day 1 to day N. "
                 "Currently, we support N being [1, 3, 5, 7].",
                 style=styles['info']),
        dcc.Input(id='time-stamps', type='text', style=styles['input'], value=''),
    ]),

    html.Div([
        html.Label("Number of Predicting Days", style=styles['label']),
        html.Div("This is the number of MELD scores to predict.",
                 style=styles['info']),
        dcc.Input(id='num-of-predicting-days', type='text', style=styles['input'], value=''),
    ]),

    html.Button('Submit', id='submit-button', n_clicks=0, style=styles['button'], className='button'),

    html.Div(id='output-container', style=styles['output'], children=[
        html.Div("The predicted MELD scores will be displayed here.", style=styles['info'])
    ])
])

@app.callback(
    Output('output-container', 'children'),
    Input('submit-button', 'n_clicks'),
    State('meldna-score', 'value'),
    State('time-stamps', 'value'),
    State('num-of-predicting-days', 'value')
)
def update_output(n_clicks, meld_na_scores, time_stamps, num_of_predicting_days):
    if n_clicks > 0:
        # Example:
        import subprocess
        result = subprocess.run(['python', '-m', 'pkgs.webapp.predict', meld_na_scores, time_stamps, num_of_predicting_days],
                                capture_output=True, text=True)
        return result.stdout

    return html.Div("The predicted MELD scores will be displayed here.", style=styles['info'])

# Add CSS for button hover effect
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .button:hover {
                background-color: #45a049 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=10000)
