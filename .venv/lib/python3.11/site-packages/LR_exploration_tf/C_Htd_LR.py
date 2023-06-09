from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np
import math

class C_Htd_LR(Callback):

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(C_Htd_LR, self).__init__()
        self.U = 3
        self.L = -3
        self.total_epoch = 10
        self.lr_cg_th = -1
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.clr_max_lr = base_lr*1.2
        self.clr_min_lr = base_lr*0.8
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.htd_iterations = 0.
        self.htd_trn_iterations = 0.
        self.history = {}
        self.iter = 0
        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def htdlr(self):
        inter_var1 = self.base_lr
        inter_var2 = (self.base_lr + self.max_lr)/2
        # t_by_T = (self.clr_iterations-self.lr_cg_th)/(self.total_epoch - self.lr_cg_th)
        t_by_T = self.htd_iterations/self.total_epoch
        inter_var3 = 1-math.tanh(self.L + (self.U - self.L)*t_by_T)
        print("clr_iter : ", self.htd_iterations)
        return (inter_var1 + inter_var2*inter_var3)

    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.clr_min_lr + (self.clr_max_lr-self.clr_min_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.clr_min_lr + (self.clr_max_lr-self.clr_min_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}
        self.iter +=1
        # print("iter : ", self.iter)
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.max_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.htdlr())  
          
    def on_batch_end(self, batch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.htd_iterations += 1
        self.htd_trn_iterations += 1
        # print("epoch : ", epoch)
        self.clr_max_lr = self.htdlr()*1.2
        self.clr_min_lr = self.htdlr()*0.8