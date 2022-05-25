#!/usr/bin/python
from flask import Flask
from flask_restplus import Api, Resource, fields
import joblib
from m09_model_deployment import clasificar_pelicula
from m09_model_deployment import split_into_lemmas


# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Movies Classifier API',
    description='Movies Classifier API')

ns = api.namespace('predict', 
     description='Movies Classifier')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Movie''s Year', 
    location='args')

parser.add_argument(
    'Title', 
    type=str, 
    required=True, 
    help='Movie''s Title', 
    location='args')

parser.add_argument(
    'Plot', 
    type=str, 
    required=True, 
    help='Movie''s Plot', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})


@ns.route('/')
class MovieApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": clasificar_pelicula(args['Year'], args['Title'], args['Plot'] )
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
