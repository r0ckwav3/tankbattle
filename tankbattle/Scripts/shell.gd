class_name Shell
extends CharacterBody2D

const my_scene = preload("res://Scenes/shell.tscn")

var SPEED = 400

func _on_ready():
	$Life.start()

static func new_shell(spawnPos: Vector2, spawnRot: float) -> Shell:
	var instance: Shell = my_scene.instantiate()
	instance.global_position = spawnPos
	instance.global_rotation = spawnRot
	return instance

func _physics_process(delta: float) -> void:
	velocity = -Vector2(cos(global_rotation), sin(global_rotation)) * SPEED
	$shell.play("default")
	move_and_slide()

func _on_life_timeout() -> void:
	# not working
	queue_free()
