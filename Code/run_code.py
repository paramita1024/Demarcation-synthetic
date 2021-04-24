import getopt 
import time
#import matplotlib.pyplot as plt
import numpy as np
from myutil import *
from slant import *
from cherrypick_a import *
from cherrypick_d import *
from cherrypick_e import *
from cherrypick_t import *


def parse_command_line_input():

	argv = sys.argv[1:]
	opts, args = getopt.getopt(argv, 'p:t:m:f:l:', ['path_to_file','time','method', 'frac', 'lamb' ])

	

	path_to_file=''
	time = 0.0
	method = 'cherrypick_a'
	frac = 0.8
	lamb = 1.0

	for opt, arg in opts:

		if opt == '-p':
			path_to_file=arg

		if opt == '-m':
			method = arg
			
		if opt == '-t':
			time = float(arg)
			
		if opt == '-f':
			frac = float(arg) 

		if opt == '-l':
			lamb=float(arg)

	return path_to_file, method, time, frac, lamb



# def f( data, method, param, frac, time, fr_threshold):

# 	# input :  data , lamb , required param , time 

# 	# subset sel 

# 	# slant 

# 	return acc, fr

# def exp1( data, method, fr_threshold):

# 	l_acc = []
# 	l_fr = [] 

# 	for t in [ 0., 0.1, 0.2, 0.3, 0.4, 0.5 ]:

# 		acc, fr = f( data, method, param, 0.8, t , fr_threshold)

# 		l_acc.append( acc )
# 		l_fr.append( fr )



	

# def f1():

	# input : file , expname , method 

	# find param  , go to required setting  , call for method , 

	# return an array of acc and FR , print it 

	# similar arrangement for synthetic too



def get_fr( pred, target, fr_threshold): 


	def get_polarity( s,thres):
		polarity_s=np.zeros(s.shape[0])
		polarity_s[np.where(s>thres)[0]]=1
		polarity_s[np.where(s<-thres)[0]]=-1
		return polarity_s

	num_test=target.shape[0]
	polar_pred=get_polarity( pred, fr_threshold)
	polar_true=get_polarity( target, fr_threshold)

	return float(np.count_nonzero(polar_true-polar_pred))/num_test

def get_mse(s,t):
	return np.mean((s-t)**2)


def get_subset(  data, frac, lamb, method , w=None, v=None  ):


	slant_obj = slant( obj= data , init_by = 'dict'  , data_type = 'real', tuning = True, tuning_param = [w,v,lamb] ) 
	sigma = slant_obj.get_sigma_covar_only()
	
	start = time.time()
	num_sel_msg = int( frac * data['train'].shape[0] )


	if method == 'cherrypick_a':
		cherrypick_obj = cherrypick_a( obj = data, sigma_covariance = sigma, lamb =  lamb, w=w ) 
		subset = cherrypick_obj.demarkate_process( frac = frac )

	if method == 'cherrypick_d':
		cherrypick_obj = cherrypick_d( obj = data, sigma_covariance = sigma, lamb =  lamb, w=w ) 
		subset = cherrypick_obj.demarkate_process( frac = frac )

	if method == 'cherrypick_e':
		cherrypick_obj = cherrypick_e( obj = data, sigma_covariance = sigma, lamb =  lamb, w=w ) 
		subset = cherrypick_obj.demarkate_process( frac = frac )

	if method == 'cherrypick_t':
		cherrypick_obj = cherrypick_t( obj = data, sigma_covariance = sigma, lamb =  lamb, w=w ) 
		subset = cherrypick_obj.demarkate_process( frac = frac )

	if method == 'robust_ht':
		obj = robust_ht( obj = data , init_by = 'dict', lamb = lamb , w_slant=w ) 
		obj.initialize_data_structures()
		tmp_w, subset, tmp_norm_of_residual = cherrypick_obj.robust_regression_via_hard_threshold( method = 'FC', max_itr = 50 , frac_end = frac) 
		
		
	if method == 'slant':
		subset =  np.ones( data['train'].shape[0], dtype = bool) 

	return subset

	
def get_slant_res( data, subset, lamb, t, fr_threshold , w, v, flag_return_res=False , int_gen = 'Hawkes' ):
	
	full_train =  np.copy( data['train'] )	
	data['train'] = data['train'][ subset , :]
	
	slant_obj=slant( obj=data, init_by='dict', data_type='real', tuning_param=[ w, v, lamb] , int_generator= int_gen)
	slant_obj.estimate_param()
	slant_obj.set_train(full_train)

	if t==0:
		result_obj = slant_obj.predict( num_simulation=1, time_span_input = t )
	else:
		result_obj = slant_obj.predict( num_simulation= 20, time_span_input=t )

	fr = 0 # get_fr( np.average(result_obj['predicted'], axis=1).flatten(), result_obj['true_target'].flatten(), fr_threshold)

	if flag_return_res:
		return result_obj['MSE'], fr , result_obj

	else:
		return result_obj['MSE'], fr 


def get_method_res( path_to_file, method, frac, t, lamb ):

	file_name=path_to_file.split('/')[-1].split('_n_')[0]
	print( file_name)
	print(load_data('w_v_synthetic').keys())
	# exit()
	w=load_data('w_v_synthetic')[file_name]['w']
	v=load_data('w_v_synthetic')[file_name]['v']

	# data_file = '../Data/' + file_name 
	data = load_data(path_to_file)
	# data = {'nodes': data_all['nodes'], 'edges': data_all['edges'] , 'train': data_all['all_user']['train'] , \
	# 	'test': data_all['all_user']['test']  }
	data['train'] = np.copy( data['all_user']['train'])
	data['test'] = np.copy( data['all_user']['test'])
	
	# exit()
	fr_threshold = 0 #float(load_data('hyper_params')['threshold_fr'][file_name])

	subset = get_subset( data, frac, lamb, method, w=w, v=v )

	mse,_ = get_slant_res( data, subset, lamb, t, fr_threshold, w, v)
	
	return mse

			

def main():

	
	path_to_file, method, lamb, frac, time = parse_command_line_input()

	# exit()
	mse, failure_rate = get_method_res( path_to_file, method, frac=frac, t=time, lamb=lamb)
			
	print( 'MSE:', mse )

	# print(' Failure Rate:', failure_rate)




if __name__== "__main__":
  main()



