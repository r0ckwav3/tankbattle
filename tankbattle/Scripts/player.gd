extends CharacterBody2D

var TURN_SPEED = 0.02
var MOVE_SPEED = 2
var BARREL_LENGTH = 230

# polar coordinates
@onready var movement_axis = Vector2.ZERO
@onready var scene = get_parent()

var shot_cooldown = false

func _physics_process(delta):
	$turret.look_at(get_global_mouse_position())
	$turret.rotate(PI)
	move(delta)
	if (Input.is_action_pressed("shoot")) and not shot_cooldown:
		shoot()	

func move(delta):
	movement_axis.x = int(Input.is_action_pressed("move_forward")) - int(Input.is_action_pressed("move_backward"))
	var old_dir = movement_axis.y
	movement_axis.y += (int(Input.is_action_pressed("turn_right")) - int(Input.is_action_pressed("turn_left"))) * TURN_SPEED
	var delta_rotation = movement_axis.y - old_dir

	if (movement_axis.x or delta_rotation):
		$body.play("default")
	else:
		$body.stop()

	position.x += cos(movement_axis.y) * movement_axis.x * MOVE_SPEED
	position.y += sin(movement_axis.y) * movement_axis.x * MOVE_SPEED

	rotate(delta_rotation)

func shoot():
	var total_rotation = $turret.rotation + movement_axis.y
	var pos_offset = Vector2(cos(total_rotation), sin(total_rotation)) * BARREL_LENGTH
	var instance = Shell.new_shell(global_position - pos_offset, total_rotation)
	scene.add_child.call_deferred(instance)
	shot_cooldown = true
	$ShotCooldown.start()

func _on_timer_timeout() -> void:
	shot_cooldown = false
