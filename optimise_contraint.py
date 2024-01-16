from scipy.optimize import minimize, Bounds
import numpy as np
import matplotlib.pyplot as plt
'''
get delta2 from delta1 by minimising the contraint
'''


def get_init_deltas(contraint_func, data_info, _print=True, _plot=True):
   '''
   contraint_func: to equal 0 with args: delta1, delta2, data_info
   delta1: fixed delta1 value
   '''
   if _print == True:
      print('-----------')
      print(f'optimising init deltas wrt. contraint')
   
   # initial guess of deltas - use 1 as gives us the best chance
   deltas_init = (np.random.uniform(), np.random.uniform())
   deltas_init = (1, np.random.uniform())


   def loss_wrapper(deltas_init):
      return np.abs(contraint_func(deltas_init[0], deltas_init[1], data_info))


   # sample loss function to find rough min estimate
   d2 = np.linspace(0, 1, 10000)
   loss = np.abs(contraint_func(deltas_init[0], d2, data_info))

   if _print == True:
      print(f'min: {np.min(loss)}')

   # find min via optimisation
   res = minimize(loss_wrapper,
                  deltas_init,
               #    (data_info),
                  #    method='SLSQP',
                  bounds=Bounds([0, 0], [1, 1]),
               #    jac=use_grad,  # use gradient
               #    constraints=contrs
                  )
   
   if _print == True:
      print(f'deltas init found: {res.x[0]}, {res.x[1]}')
      print(f'contraint {contraint_func(res.x[0], res.x[1], data_info)}')
      print('-----------')

   if _plot == True:
      plt.plot(d2, loss)
      plt.show()

   return res.x[0], res.x[1]
