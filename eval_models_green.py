from model_general_nn import GeneralNN
import run_nn_green
import math
import os
from collections import OrderedDict
import torch
import pickle


def print_best(ordered_dict, num):
  for i,(model,score) in enumerate(list(ordered_dict.items())):
    print((i + 1), ") ", model, "\t", score)
    if i == num:
      return

widths        = [50,100,150,200,250,300]
depths        = [1,2,3,4]
epochs        = 800
epoch_step    = 2
batch         = 64
learning_rate = 5e-7
prob = True


data_name  = 'green'



model_dict = {}

for width in widths:
  for depth in depths:
    
    print("------------------------------Evaluating Depth=", depth, "\tWidth=", width, "------------------------------")
    
    
    #best_loss  = math.inf
    #best_model = ''
    for i in range(epoch_step, epochs, epoch_step):
      dir_str = str('_models/sweep/')
      info_str = "w-" + str(width) + "_d-" + str(depth) + "_e-" + str(i+epoch_step) + "_lr-" + str(learning_rate) + "_b-" + str(batch) + "_ds-" + str(data_name) + "_p-" + str(prob)
      model_name = dir_str + info_str + ".pth"
      if os.path.exists(model_name):
        model = torch.load(model_name)
      else:
        print("Couldn't find: ", model_name, "! Continuing...")
        continue
      loss = run_nn_green.eval(model)
      model_dict[info_str] = loss
      print("Loss for ", info_str, " = ", loss)
      #if loss < best_loss:
      #  best_loss  = loss
      #  best_model = model_name   
    print("------------------------------Completed Depth=" + str(depth) + "\tWidth=", str(width), "!------------------------------")

a_acc_x = OrderedDict(sorted(model_dict.items(), key=lambda t: t[1][0]))
a_acc_y = OrderedDict(sorted(model_dict.items(), key=lambda t: t[1][1]))
a_acc_z = OrderedDict(sorted(model_dict.items(), key=lambda t: t[1][2]))
pitch   = OrderedDict(sorted(model_dict.items(), key=lambda t: t[1][3]))
roll    = OrderedDict(sorted(model_dict.items(), key=lambda t: t[1][4]))
overall = OrderedDict(sorted(model_dict.items(), key=lambda t: t[1]))



print("\n\n------------------------- RESULTS -------------------------\n\n")
print("\n------------------------- Accel X -------------------------")
print_best(a_acc_x, 25)
print("\n------------------------- Accel Y -------------------------")
print_best(a_acc_y, 25)
print("\n------------------------- Accel Z -------------------------")
print_best(a_acc_z, 25)
print("\n------------------------- Pitch -------------------------")
print_best(pitch,   25)
print("\n------------------------- Roll -------------------------")
print_best(roll,    25)
print("\n------------------------- Overall -------------------------")
print_best(overall, 25)


with open("greenresults.pkl", 'wb') as pickle_file:
  pickle.dump((a_acc_x,a_acc_y,a_acc_z,pitch,roll), pickle_file, protocol=2)





    
