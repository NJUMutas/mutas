from flask import Flask, request
import json
from mutas_scheduler.core.mutas_solver import MutasSolver
from mutas_scheduler.data.global_data import GlobalData

app = Flask(__name__)

ms = MutasSolver()
gd = GlobalData()


@app.route("/mutas/echo")
def init():
    return json.dumps({
        'message': 'mutas is running successfully'
    }, ensure_ascii=False)


@app.route("/mutas/updateGd", methods={'GET', 'POST'})
def updateGd():
    data = request.args.to_dict()
    try:
        gd.update(data)
        return json.dumps({
            'message': 'mutas update suceessfully',
            'data': data
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            'message': 'mutas update failed, please check your data'
        }, ensure_ascii=False)


@app.route("/mutas/commitUserTask", methods={'POST'})
def commitUserTask():
    data = request.args.to_dict()
    return json.dumps({
        'message': 'task has been committed successfully',
        'data': data
    }, ensure_ascii=False)


if __name__ == '__main__':
    app.run()
