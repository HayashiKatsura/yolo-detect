import logging
from flask import Blueprint
from flask_cors import CORS

api_files = Blueprint('api_files', __name__)
CORS(api_files)

from . import routes
