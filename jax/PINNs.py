import jax
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
import jaxlib
import optax
from tensorflow_probability.substrates import jax as tfp
from functools import partial

from time import time





class param_vec_jax():
    def __init__(self,params):
        param_leafs, self.params_tree = jax.tree_util.tree_flatten(params)
        self.param_lens = []
        self.param_shapes = []
        for p in param_leafs:
            ps = jnp.array(p.shape)
            
            self.param_lens.append(jnp.prod(ps))
            
            self.param_shapes.append(p.shape)
        print(len(self.param_lens),len(self.param_shapes))
       
        
        
    def param_to_vector(self,params):
        param_leafs, _ = jax.tree_util.tree_flatten(params)
        return jnp.concatenate([p.flatten() for p in param_leafs])
        
         
    def vector_to_param(self,param_vec):
        cum = 0
        leafs = []
        for l,s in zip(self.param_lens,self.param_shapes):
            
            parleaf = jax.lax.dynamic_slice(param_vec,[cum],[l])
            
            leafs.append(parleaf.reshape(s))
            cum = cum+l
        return jax.tree_util.tree_unflatten(self.params_tree, leafs)
        




class ExplicitMLP(nn.Module):
    features: Sequence[int]
    lb : jaxlib.xla_extension.ArrayImpl
    ub : jaxlib.xla_extension.ArrayImpl

    def setup(self):
    # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
    # for single submodules, we would just write:
    # self.layer1 = nn.Dense(feat1)

    def __call__(self, inputs):
       # print(ub,lb)
        x = 2.0*(inputs-self.lb)/(self.ub-self.lb) - 1.0
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.tanh(x)
        return x


class PINN():
    def __init__(self,dom_bounds,prob_coef,DTYPE='float32'):
        self.DTYPE = DTYPE
        self.lwb = jnp.array(dom_bounds[0],dtype=DTYPE)
        self.upb = jnp.array(dom_bounds[1],dtype=DTYPE)
        self.dim = len(dom_bounds[0])
        
        self.param_vec: param_vec_jax
        
        
            
        self.prob_coef = jnp.array(prob_coef,dtype=DTYPE)
        
        print("Prob coef",self.prob_coef[0])
        print("Bounds",self.lwb,self.upb)
        
        
        self.model: ExplicitMLP

    def pt(self):
        print(self.model.summary())
        
    

        
        
        
    def sample_domain(self,key,n=10):
        x = []
        for i in range(self.dim):
            print(self.lwb[i],self.upb[i])
            x.append(random.uniform(key,[n],minval=self.lwb[i],maxval=self.upb[i]))
            
        return x
    
    
    def make_model(self,no_hid_layers,no_of_neu=20,act='tanh'):
        features = []
        for i in range(0,no_hid_layers):
            features.append(no_of_neu)
        features.append(1)

        return ExplicitMLP(features,self.lwb,self.upb)
    
    
    
    
    
    def diff_eqn(self,params,dom_p):
        t = dom_p[0]
        x = dom_p[1]

        def pred(params,t,x):
            out_e = jnp.eye(1,1,0).flatten()
            xx = jnp.eye(1,2,0)*t + jnp.eye(1,2,1)*x
            f = self.model.apply(params,xx).flatten()
            return jnp.inner(f,out_e)
    

        dx_pred = jax.grad(pred,argnums=2)
        dxx_pred = jax.grad(dx_pred,argnums=2)
        dt_pred = jax.grad(pred,argnums=1)
    
    
    
        eqn = dt_pred(params,t,x) + pred(params,t,x)*dx_pred(params,t,x)-(0.01/jnp.pi)*dxx_pred(params,t,x)
    
        return jnp.inner(eqn,eqn)/2.0



 
    
    
    def compute_loss(self,params,dom,bndry,f_bndry):
    
        def sq_loss(x,y):
            yp = self.model.apply(params,x)
            return jnp.inner((y-yp),(y-yp))/2.0
        def apply_diff_eq(dom):
            return self.diff_eqn(params,dom)
    
        return jnp.mean(jax.vmap(sq_loss)(bndry,f_bndry),axis=0) + jnp.mean(jax.vmap(apply_diff_eq)(dom),axis=0)
    
    
    
    def ready_model(self,key,optim):
        x = jnp.reshape(self.lwb,(1,self.dim))
        self.ini_params = self.model.init(key,x)
        self.loss_and_grad = jax.value_and_grad(self.compute_loss)
        self.optim = optim


    def ready_model_vec(self,key,dom_points,bndry_points,f_bndry):

        params = self.model.init(key,bndry_points)
        self.param_vec_func = param_vec_jax(params)
        self.param_vec = self.param_vec_func.param_to_vector(params)
        
        self.dom_points = dom_points
        self.bndry_points = bndry_points
        self.f_bndry  = f_bndry

        self.loss_and_grad_vec = self.get_loss_and_grad_vec()


    @partial(jax.jit, static_argnums=(0,))
    def loss_and_grad_func(self,params,dom,bndry,f_bndry):
        return self.loss_and_grad(params,dom,bndry,f_bndry)
    

    
    def loss_and_grad_V(self,param_vector):
        params = self.param_vec_func.vector_to_param(param_vector)
        return self.compute_loss(params,self.dom_points,self.bndry_points,self.f_bndry)
    
    def get_loss_and_grad_vec(self):
        return jax.value_and_grad(self.loss_and_grad_V)
    
    @partial(jax.jit, static_argnums=(0,))
    def loss_and_grad_vec_func(self,params):
        return self.loss_and_grad_vec(params)


    def train(self,dom,bndry_points,f_bndry,epochs=10,print_each_n=10,params=None,optstate=None):
        if optstate is None:
            optstate = self.optim.init(self.ini_params)
        if params is None:
            params = self.ini_params
        params_best = self.ini_params
        loss_min = 1e10     

        for i in range(epochs):
            #t,x=get_domain_points(key2,n=100)
            #key1,key2 = random.split(key2)
            loss,grd = self.loss_and_grad_func(params,dom,bndry_points,f_bndry)
            updates,optstate = self.optim.update(grd,optstate)
            params = optax.apply_updates(params,updates)
            if loss_min>loss:
                params_best = params
                loss_min = loss
            
            if i % print_each_n == 0:
                print('Loss step {}: '.format(i), loss)
        
        print("best loss ",loss_min)

        return params_best,loss_min

    
   
    
    
   
    
    
    

    def batch_train(self,dom_points,bndry_points,f_bndry,epochs = 5000,batch_n=32,params=None,optstate=None):

        if optstate is None:
            optstate = self.optim.init(self.ini_params)
        if params is None:
            params = self.ini_params
        params_best = self.ini_params
        loss_min = 1e10 

        N = bndry_points.shape[0]
        n_batches = int(N/batch_n)
        
        print(n_batches,batch_n)
        
        losses = []
        t0 = time()
        for i in range(epochs):
            loss_ep=0.0
            for j in range(n_batches):
                
                loss,grd = self.loss_and_grad_func(params,dom_points[j*batch_n:(j+1)*batch_n,:],bndry_points[j*batch_n:(j+1)*batch_n,:],f_bndry[j*batch_n:(j+1)*batch_n])
                loss_ep = loss_ep+loss
                updates,optstate = self.optim.update(grd,optstate)
                params = optax.apply_updates(params,updates)

            losses.append(loss_ep)
    
            if i%50==0:
                print('It {:05d}: loss = {:10.8e}'.format(i,loss_ep/n_batches))

            if loss_min>loss_ep:
                params_best = params
                loss_min = loss
        return params_best,loss_min
    
    def train_tfp_bfgs(self,key,dom_points,bndry_points,f_bndry,params_vec=None,tolerance=1e-08,
                                                            x_tolerance=0,
                                                            f_relative_tolerance=0,
                                                            initial_inverse_hessian_estimate=None,
                                                            max_iterations=50,
                                                            parallel_iterations=1,
                                                            stopping_condition=None,
                                                            validate_args=True,
                                                            max_line_search_iterations=50,
                                                            f_absolute_tolerance=0,
                                                            name=None):

        self.ready_model_vec(key,dom_points,bndry_points,f_bndry)
        if params_vec is None:
            params_vec = self.param_vec


        

        res=tfp.optimizer.bfgs_minimize(self.loss_and_grad_vec_func,params_vec,tolerance,
                                                                        x_tolerance,
                                                                        f_relative_tolerance,
                                                                        initial_inverse_hessian_estimate,
                                                                        max_iterations,
                                                                        parallel_iterations,
                                                                        stopping_condition,
                                                                        validate_args,
                                                                        max_line_search_iterations,
                                                                        f_absolute_tolerance,
                                                                        name=None)
        
        return res
    

    def train_tfp_lbfgs(self,key,dom_points,bndry_points,f_bndry,params_vec=None,previous_optimizer_results=None,
                                                                    num_correction_pairs=10,
                                                                    tolerance=1e-08,
                                                                    x_tolerance=0,
                                                                    f_relative_tolerance=0,
                                                                    initial_inverse_hessian_estimate=None,
                                                                    max_iterations=50,
                                                                    parallel_iterations=1,
                                                                    stopping_condition=None,
                                                                    max_line_search_iterations=50,
                                                                    f_absolute_tolerance=0,
                                                                    name=None):
        

        self.ready_model_vec(key,dom_points,bndry_points,f_bndry)
        if params_vec is None:
            params_vec = self.param_vec

        res=tfp.optimizer.lbfgs_minimize(self.loss_and_grad_vec_func ,params_vec,previous_optimizer_results,
                                                                    num_correction_pairs,
                                                                    tolerance,
                                                                    x_tolerance,
                                                                    f_relative_tolerance,
                                                                    initial_inverse_hessian_estimate,
                                                                    max_iterations,
                                                                    parallel_iterations,
                                                                    stopping_condition,
                                                                    max_line_search_iterations,
                                                                    f_absolute_tolerance,
                                                                    name)
        
        return res