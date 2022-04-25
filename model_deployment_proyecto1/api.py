#!/usr/bin/python
from flask import Flask
from flask_restplus import Api, Resource, fields
import joblib
from m09_model_deployment import predict_price

# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Price Prediction API',
    description='Price Prediction API')

ns = api.namespace('predict', 
     description='Price Regressor')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Model Year', 
    location='args')

parser.add_argument(
    'Mileage', 
    type=int, 
    required=True, 
    help='Car Mileage', 
    location='args')

parser.add_argument(
    'State', 
    type=str, 
    required=True, 
    help='Registered State', 
    location='args')

parser.add_argument(
    'Make', 
    type=str, 
    required=True, 
    help='Brand', 
    location='args')

parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='Car Model', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

# Definición de la clase para disponibilización
@ns.route('/')
class PriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_price(args['Year'], args['Mileage'], args['State'], args['Make'], args['Model'] )
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
