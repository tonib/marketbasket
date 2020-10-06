import flask
from predict import Prediction
from transaction import Transaction

app = flask.Flask(__name__)
#app.config["DEBUG"] = True

predictor = Prediction()

@app.route('/predict', methods=['POST'])
def predict():
    content = flask.request.get_json(force=True)
    #print(content)

    t = Transaction.from_labels( content['item_labels'] , content['customer_label'] )
    #print(t)

    prediction = predictor.predict_single( t, content['n_results'] )
    #print(prediction)

    return flask.jsonify( [prediction[0].tolist() , prediction[1].tolist()] )

app.run(threaded=False)
