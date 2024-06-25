# run_app.py

from pkgs.webapp.main import app

if __name__ == "__main__":
    app.run_server()

# This is required by Gunicorn to serve the application
application = app.server
