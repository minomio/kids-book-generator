# https://www.youtube.com/watch?v=GMppyAPbLYk
# https://www.youtube.com/watch?v=XrG_TlwPtsU

from flask import Flask
from flask_restful import Api, Resource
import restful_sub
import threading

app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    def get(self, prompt):
        a = restful_sub.print_something()
        threading.Thread(target=restful_sub.upload_pic(prompt)).start()
        return {"data": prompt, "a": a}

api.add_resource(HelloWorld, "/helloworld/<string:prompt>")

if __name__ == "__main__":
    app.run(debug=True)
