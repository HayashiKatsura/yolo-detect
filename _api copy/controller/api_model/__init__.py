import logging
from flask import Blueprint
from flask_cors import CORS

api_model = Blueprint('api_model', __name__)
CORS(api_model)

from . import routes
