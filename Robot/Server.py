from flask import Flask, request, jsonify
from Robot import Robot

app = Flask(__name__)
robot = Robot()

@app.route('/')
def home():
    return "EV3 Flask Server Running"

@app.route('/move_forward', methods=['POST'])
def move_forward():
    data = request.json
    distance = data.get('distance', 10)  # Default 10 cm
    speed = data.get('speed', 50)  # Default speed 50
    robot.move_forward(distance, speed)
    return jsonify({"status": "moved forward", "distance": distance, "speed": speed, "position": robot.get_position()})

@app.route('/move_backward', methods=['POST'])
def move_backward():
    data = request.json
    distance = data.get('distance', 10)
    speed = data.get('speed', 50)
    robot.move_backward(distance, speed)
    return jsonify({"status": "moved backward", "distance": distance, "speed": speed, "position": robot.get_position()})

@app.route('/turn_left', methods=['POST'])
def turn_left():
    data = request.json
    angle = data.get('angle', 90)  # Default 90 degrees
    speed = data.get('speed', 30)
    robot.turn_left(angle, speed)
    return jsonify({"status": "turned left", "angle": angle, "speed": speed, "new_angle": robot.get_angle()})

@app.route('/turn_right', methods=['POST'])
def turn_right():
    data = request.json
    angle = data.get('angle', 90)
    speed = data.get('speed', 30)
    robot.turn_right(angle, speed)
    return jsonify({"status": "turned right", "angle": angle, "speed": speed, "new_angle": robot.get_angle()})

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "position": robot.get_position(),
        "angle": robot.get_angle()
    })

@app.route('/get_position', methods=['GET'])
def get_position():
    return jsonify({
        "position": # robot.get_position()
        robot.get_position()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)