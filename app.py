# Loading the dependencies
from flask import Flask,request,jsonify
from flask_cors import CORS
import rec
app = Flask(__name__)
CORS(app)  
     

@app.route('/', methods=['GET'])
def recommend_movies():
    res=rec.recomended_movies(request.args.get('title'))
    return jsonify(res)


if __name__=='__main__':
     app.run(port = 5000, debug = True)
