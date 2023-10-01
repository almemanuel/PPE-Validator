from flask import Flask, request, jsonify

try:
    from ia import search_items
except:
    from script.ia import search_items
import json


app = Flask(__name__)

@app.route('/search', methods=['GET'])
def search():
    try:
        img = request.get_json()['img']
        print(f'img: {img}')
        img = search_items(img)
        return jsonify({'msg': img}), 200
    except Exception as e:
        return jsonify({'msg': f'{e}'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)