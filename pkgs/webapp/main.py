# app.py

import dash
import dash.html as html

app = dash.Dash(__name__)

app.layout = html.Div('Hello, World!')

if __name__ == '__main__':
    app.run_server(debug=True)
