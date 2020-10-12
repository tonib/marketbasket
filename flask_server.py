import flask
from predict import Prediction
from transaction import Transaction

app = flask.Flask(__name__)
#app.config["DEBUG"] = True

predictor = Prediction()

@app.route('/v1/models/basket:predict', methods=['POST'])
def predict():

    try:
        content = flask.request.get_json(force=True)
        #print(content)

        inputs = content['inputs']
        t = Transaction.from_labels( inputs['item_labels'] , inputs['customer_label'] )
        #print(t)

        prediction = predictor.predict_single( t, inputs['n_results'] )
        #print(prediction)

        # astype is required in Windows, otherwise it throws "TypeError: Object of type bytes is not JSON serializable"
        prediction = { 'outputs': { 'output_0': prediction[0].astype('U').tolist() , 'output_1': prediction[1].tolist() } }
        return flask.jsonify( prediction )
    except Exception as err:
        print(err)
        return flask.jsonify( { 'error': str(err) } )

@app.route("/hello")
def hello():
    return "Hello world!"

if __name__=='__main__':
    app.run(threaded=False, host='0.0.0.0', port=5001)