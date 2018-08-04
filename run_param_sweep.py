from model_general_nn import GeneralNN
import run_nn
import run_nn_green

widths        = [50,100,150,200,250,300]
depths        = [1,2,3,4]
epochs        = 800
epoch_step    = 2
batch         = 64
learning_rate = 5e-7
prob = True


for width in widths:
  for depth in depths:
    print("------------------------------Running Depth=", depth, "\tWidth=", width, "------------------------------")
    try:
      model = GeneralNN(n_in_input = 4, n_in_state = 5, hidden_w=width, n_out = 5, state_idx_l = [0,1,2,3,4], prob=prob, pred_mode='Next State', depth = depth) 
      run_nn.run(model, learning_rate, epochs, epoch_step, batch, width, depth)
    except:
      print("failed on depth: ", depth, " width: ", width)
    print("------------------------------Completed Depth=" + str(depth) + "\tWidth=", str(width), "!------------------------------")

  
print("Pink Sweep Complete!")


for width in widths:
  for depth in depths:
    print("------------------------------Running Depth=", depth, "\tWidth=", width, "------------------------------")
    try:
      model = GeneralNN(n_in_input = 4, n_in_state = 5, hidden_w=width, n_out = 5, state_idx_l = [0,1,2,3,4], prob=prob, pred_mode='Next State', depth = depth) 
      run_nn_green.run(model, learning_rate, epochs, epoch_step, batch, width, depth)
    except:
      print("failed on depth: ", depth, " width: ", width)

    print("------------------------------Completed Depth=" + str(depth) + "\tWidth=", str(width), "!------------------------------")

print("Green Sweep Complete!")


import eval_models
import eval_models_green