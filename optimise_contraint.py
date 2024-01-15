from scipy.optimize import minimize, Bounds
import numpy as np
'''
get delta2 from delta1 by minimising the contraint
'''


def get_init_deltas(contraint_func, data_info):
   '''
   contraint_func: to equal 0 with args: delta1, delta2, data_info
   delta1: fixed delta1 value
   '''
   deltas_init = (np.random.uniform(), np.random.uniform())
   deltas_init = (1, np.random.uniform())


   def loss_wrapper(deltas_init):
      return np.abs(contraint_func(deltas_init[0], deltas_init[1], data_info))

   import matplotlib.pyplot as plt
   d2 = np.linspace(0, 1, 100)
   loss = np.abs(contraint_func(deltas_init[0], d2, data_info))
   print(f'min: {np.min(loss)}')
   plt.plot(d2, loss)
   plt.show()

   res = minimize(loss_wrapper,
                  deltas_init,
               #    (data_info),
                  #    method='SLSQP',
                  bounds=Bounds([0, 0], [1, 1]),
               #    jac=use_grad,  # use gradient
               #    constraints=contrs
                  )
   print(res.x[0], res.x[1])

   print(contraint_func(res.x[0], res.x[1], data_info))

   d2 = np.linspace(0, 1, 100)
   loss = np.abs(contraint_func(deltas_init[0], d2, data_info))
   print(f'min: {np.min(loss)}')
   print('-----------')

   return res.x[0], res.x[1]
