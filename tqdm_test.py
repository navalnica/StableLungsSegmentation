from matplotlib import pyplot as plt
import numpy as np
import os

#
# x = np.arange(-10, 10)
# fig, ax = plt.subplots(1, 2)
# ax[0].plot(x, x)
# ax[1].plot(x, x ** 3)
# print(f'display available: {os.environ["DISPLAY"]}')
# plt.show()

# r = os.system('python -c "import matplotlib.pyplot as plt;plt.figure()"')
# if r != 0:
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     fig.savefig('myfig.png')
# else:
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     plt.show()

# plt.show()
#
# from time import sleep
# import tqdm
#
#
# for x in range(2):
#     print(f'x: {x}')
#
#     gen1 = (x ** 2 for x in range(10))
#     gen2 = (-x for x in range(5))
#
#
#     with tqdm.tqdm_notebook(total=30, desc='first', unit='unit') as pbar1:
#         for e_ix, x in enumerate(gen1, start=1):
#             sleep(0.1)
#             # if e_ix > 5: break
#             pbar1.update(3)
#
#     for e_ix, x in tqdm.tqdm(enumerate(gen2, start=1), total=10, desc='second', unit='unit'):
#         sleep(0.1)
#         # tqdm.tqdm.write(f'{e_ix}, {x}')
#         if e_ix > 5: break