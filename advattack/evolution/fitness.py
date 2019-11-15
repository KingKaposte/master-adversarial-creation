import numpy as np


# return the fitness of x in relation to the target
def get_fitness(sess, x, target, input_tensor, output_tensor):
    is_top_prediction = False
    output_predictions, = sess.run(output_tensor,
                             feed_dict={input_tensor: x})
    target_prediction = output_predictions[target]

    top_prediction = np.argmax(output_predictions)
    if top_prediction == target:
        is_top_prediction = True
#### if negation is active
    else:
       target_prediction = target_prediction - 1

    return target_prediction, is_top_prediction

