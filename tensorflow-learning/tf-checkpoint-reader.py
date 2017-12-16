import tensorflow as tf

reader = tf.train.NewCheckpointReader('./model/model.ckpt')

all_variables = reader.get_variable_to_shape_map()
for variable_name in all_variables:
	print(variable_name, all_variables[variable_name])

print("Value for variable v1 is ", reader.get_tensor("v1"))