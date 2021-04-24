import os
import time
from myutil import *
from spg_new import spg
# from data_preprocess import data_preprocess
import matplotlib.pyplot as plt
from create_synthetic_data import create_synthetic_data
import pickle
#from PriorityQueue import PriorityQueue
import math
import numpy.random as rnd
import sys
import numpy as np
from numpy import linalg as LA

class parameter:
	def __init__( self, mu, alpha, A, B):
		self.alpha = alpha
		self.mu = mu
		self.A = A
		self.B = B

class Grad_n_f:
	def __init__(self, coef_mat, last_coef_val, num_msg, last_time_train, v):
		self.coef_mat = coef_mat 
		self.last_coef_val = last_coef_val
		self.num_msg = num_msg
		self.last_time_train = last_time_train
		self.v = v 

	def f(self,x):
		# function value computation 
		mu = x[0]
		b = x[1:]
		if (mu == 0) and b.any() == 0 : 
			return np.exp( 20)
		# print " coef mat "
		# print self.coef_mat 

		# print "----------------"
		# print self.coef_mat.dot(b)

		# print " ----------"
		# print self.coef_mat.dot(b)  + mu

		# print "----------------"
		# print np.log( self.coef_mat.dot(b)  + mu )

		# print "-----------------"
		# print np.sum( np.log( self.coef_mat.dot(b)  + mu ) )

		
		# t1 = np.sum( np.log( self.coef_mat.dot(b)  + mu ) )
		if (self.coef_mat.dot(b)  + mu).any() == 0:
			# print "-------------------------"
			# print "mu is " + str(mu)
			# print "b is "
			# print b 
			# print "------------------------------"
			t1 = -np.exp( 20)
		else:
			t1 = np.sum( np.log( self.coef_mat.dot(b)  + mu ) )
			# print "t1 is "+ str(t1)
		# print "--------"
		# print t1


		t2 = mu*self.last_time_train - ( b.dot(self.last_coef_val - self.num_msg)) / self.v	
		# print "---- t2 -------"

		# print self.last_coef_val
		# print "-----"
		# print  self.last_coef_val - self.num_msg

		# print "-----"
		# print - ( (self.last_coef_val - self.num_msg)) / self.v	

		# print "----"
		# print - ( b.dot(self.last_coef_val - self.num_msg)) / self.v	

		# print "------"
		# print mu*self.last_time_train - ( (self.last_coef_val - self.num_msg)) / self.v	

		# print t2

		# print t2-t1

		if any(self.coef_mat.dot(b)  + mu) == 0:
			print( "f value is ",str( t2-t1))
		return ( t2 - t1 )
		
	def grad_f(self,x):
		mu = x[0]
		b = x[1:]

		# print "------------GRAD F-------------------------"
		# print self.coef_mat
		# print "---"
		# print self.coef_mat.dot(b)
		# print "--------"
		# print self.coef_mat.dot(b)  + mu
		# print "---------"
		# print np.reciprocal(self.coef_mat.dot(b)  + mu)

		recipro_term = np.reciprocal(self.coef_mat.dot(b)  + mu)


		# print "---"
		# print self.coef_mat.T
		# print "--del b t1 -"
		# print self.coef_mat.T.dot(recipro_term )
		del_b_t1 = ( self.coef_mat.T ).dot(recipro_term )
		# print "---last coef"
		# print self.last_coef_val

		# print "--"
		# print self.last_coef_val - self.num_msg
		# print "--del b t2"
		# print -(self.last_coef_val - self.num_msg) / self.v
		del_b_t2 = -(self.last_coef_val - self.num_msg) / self.v
		del_t1 = np.concatenate( ([ np.sum(recipro_term) ], del_b_t1))
		del_t2 = np.concatenate( ([self.last_time_train ], del_b_t2))
		# print "----"
		# print del_t1
		# print "---"
		# print del_t2
		grad_f = del_t2 - del_t1 
		# print "--"
		# print grad_f
		return grad_f 

class slant:
	# check object passing

	# init by 

	# alpha A mu B 

	# estimate 
	def __init__(self, init_by = None, obj = None , file = None , train = None, \
		test = None, edges = None , data_type = None, param = None , tuning = False , \
			tuning_param = None , list_of_windows = None, int_generator='Hawkes'):
		# always pass synthetic data or generate synthetic data using objects , 
		# value option is only for real data
		if data_type == 'graph':
			# we assume obj has been passed
			if init_by == 'object':
				self.edges = obj.edges
				self.num_node = self.edges.shape[ 0 ]
				self.nodes = np.arange( self.num_node )

			else:
				print( 'Initialization not defined'		)
		else:
			if init_by == None:
				print( 'Empty Initialization. Thereby creating empty object.')

			if init_by == 'values':
				self.edges = edges
				self.train = train
				self.test = test
				if data_type == 'synthetic':
					self.true_param = param
			if init_by == 'object':
				if file: 
					obj = load_data( file )
				self.edges=obj.edges

				self.train = obj.train 
				self.test = obj.test
				if data_type == 'synthetic':
					self.true_param = obj.param
				# if flag_evaluate_synthetic == True:

			if init_by=='dict':
                                #print(obj['all_user'].keys())
				self.edges=obj['edges']
				if 'train' in obj:
					self.train=obj['train']
				else:
					self.train=np.array([])
				if 'test' in obj:
					self.test=obj['test']
				else:
					self.test=np.array([])
				if data_type == 'synthetic':
					self.true_param = obj['param']

			self.num_node = self.edges.shape[0]
			self.nodes = np.arange( self.edges.shape[0])
			self.num_train= self.train.shape[0]
			self.num_test= self.test.shape[0]
			
			

			# print " TRAIN : START :  " + str(np.min( self.train[ :, 1] )) + " : END :  " + str(np.max( self.train[:,1]))
			# print " TEST : START :" + str(np.min( self.test[ :, 1] )) + " : END : " + str(np.max( self.test[:,1]))
			# print "NUMBER OF NODE : "+str( self.num_node )+ " NUMBER OF TRAIN : " + str( self.num_train ) + " : NUMBER OF TEST : " + str( self.num_test)
		if list_of_windows== None:
			list_of_windows=np.array([.04])
		self.list_of_windows = list_of_windows
		self.num_window = self.list_of_windows.shape[0]
		self.w = 10.0 # 
		self.var = 0.1 # 
		self.v = 10.0 #
		self.lambda_least_square = 0.5 # check this value too
		self.queue_length = 100
		self.max_iteration_mu_B = 1000 	
		self.int_generator=int_generator

		if tuning_param!=None :
			self.w = tuning_param[0]
			self.v = tuning_param[1]
			self.lambda_least_square = tuning_param[2]	

	def project_positive(self,v):
		return np.maximum(v,np.zeros(v.shape[0])+ sys.float_info.epsilon)

	# def Grad_f_n_f(self, coef_mat_user, last_coef_val, num_msg_user, msg_time_exp_user,  x):
	# 	# f(x) = t2(x) - t1(x)
	# 	last_time_train = self.train[-1,1] 
	# 	mu = x[0]
	# 	b = x[1:]
	# 	common_term = np.reciprocal(coef_mat_user.dot(b) * msg_time_exp_user + mu)
	# 	del_b_t1 = coef_mat_user.T.dot( common_term *  msg_time_exp_user)
	# 	del_mu_t1= np.sum(common_term)
	# 	del_b_t2 = (np.exp(self.v*last_time_train)*(last_coef_val) - num_msg_user) / self.v
	# 	del_mu_t2= last_time_train
	# 	del_t1 = np.concatenate( ([del_mu_t1], del_b_t1))
	# 	del_t2 = np.concatenate( ([del_mu_t2], del_b_t2))
	# 	# function value computation 
	# 	t1 = np.sum( np.log( coef_mat_user.dot(b) * msg_time_exp_user + mu ) )
	# 	t2 = mu + (  np.exp(self.v*last_time_train)*(b.dot(last_coef_val)) - b.dot(num_msg_user)) / self.v

	# 	grad_f = del_t2 - del_t1 
	# 	function_val = t2 - t1  
	# 	return grad_f , function_val
	def create_initial_data_matrix( self, user, index ):
		# user = 5
		neighbours = np.nonzero(self.edges[:,user].flatten())[0]
		neighbours_with_user  = np.concatenate(([user], neighbours ))

		# print "-----------edges "
		# for node in self.edges:
		# 	print node

		# print "--nbr wt user "
		# print neighbours_with_user
		# create num msg
		num_msg = np.zeros( neighbours_with_user.shape[0])
		for nbr_no in range( neighbours_with_user.shape[0]):
			num_msg[ nbr_no ] = index[ neighbours_with_user[nbr_no]].shape[0]
		last_time_train = self.train[-1,1]
		# print "last time train"
		# print last_time_train
		#  	print "num of msg " 
		#  	print num_msg
		last_coef_val = np.zeros( neighbours_with_user.shape[0])
		coef_mat = np.zeros( ( int(num_msg[0]), neighbours_with_user.shape[0] ) )
		for nbr_no in range( neighbours_with_user.shape[0]):
			nbr = neighbours_with_user[nbr_no]
			time_last = 0
			msg_index = 0
			intensity = 0  
			if nbr_no == 0 :
				# print "-------------------------------"
				for msg in self.train[ index[user],:]:
					time_curr = msg[1]
					intensity *= np.exp( -self.v*(time_curr - time_last))
					time_last = time_curr 
					coef_mat[msg_index,nbr_no] = intensity
					intensity +=1
					msg_index +=1 
					# print "msg "
					# print msg
					# print " last coef val "
					last_coef_val[nbr_no] += np.exp( - self.v*(last_time_train - time_curr ))
					# print last_coef_val[nbr_no]
					
				# print np.exp( -20)+ np.exp(-13)+np.exp(-6)
				 
			else:
				# merge msg 
				# print "-------index---------"
				index_for_both = np.sort( np.concatenate((index[user], index[nbr])) )
				# print index_for_both
				for ind in index_for_both:
					user_curr , time_curr , sentiment = self.train[ind]
					if user_curr == user:

						intensity *= np.exp(-self.v*(time_curr - time_last))
						coef_mat[msg_index, nbr_no] = intensity
						msg_index += 1 
						
					else:
						intensity = intensity*np.exp(-self.v*(time_curr - time_last))+1
						last_coef_val[nbr_no] += np.exp( - self.v*(last_time_train - time_curr ))		
					time_last = time_curr

			# # print "-----------------------"
			# print " coef mat "
			# print coef_mat 
			# print " last coef val "
			# print last_coef_val
			#--
			# t= self.train[ index[5][2],1]
			# print np.sum( np.exp( - self.v*( t - self.train[ index[3][np.arange(3)],1] )))
		return last_time_train, coef_mat, last_coef_val, num_msg , neighbours_with_user
	def find_mu_B(self, max_iter=None):
		if max_iter!=None:
			self.max_iteration_mu_B=max_iter
		# a = np.tile(np.arange(6) , 3)
		# b= np.arange(18)
		# b = b/1.7
		# c = rnd.uniform( size = 18 )
		# self.train = np.concatenate( (a,b,c) ).reshape(3,18).T

		# self.num_train = 18
		# print "----------------train -----------------------"
		# for elm in self.train:
		# 	print elm
		
		#-------------------------
		# init_function_val = np.zeros(self.num_node)
		# end_function_val =  np.zeros(self.num_node)
		#-------------------------
		mu=np.zeros(self.num_node)
		B=np.zeros((self.num_node,self.num_node))
		# #----------------------------------------DELETE -----------------------------------------------------
		# self.mu = mu
		# self.B = B
		
		# return 
		#----------------------------------------------------------------------------------------------------
		spg_obj = spg() 
		#  gradient calculation
		
		
		# plt.plot(num_msg_users)
		# plt.show()
		index={}
		for user in self.nodes : 
			index[user] = np.where(self.train[:,0]==user)[0]
		# print "----------------msg ind ----------------------------"
		# for user in self.nodes:
		# 	print index[user]
		


		for user in self.nodes: 
			# perform computation only if user has any msg 
			if index[user].shape[0] > 0 :
				last_time_train, coef_mat, last_coef_val, num_msg, neighbours_with_user  = self.create_initial_data_matrix( user, index)
				# self.mu[user], self.B[user,] =self.spectral_proj_grad()
				
				x0  = np.ones(1+ num_msg.shape[0])
				# compute parameters************
				
				# print "-------check-----------"
				# print user_msg_index
				# print "-----------------------"
				# print neighbours_with_user
				
				#-------------------------------------------
				grad_function_obj = Grad_n_f(coef_mat, last_coef_val, num_msg, last_time_train, self.v)
				# grad_function_obj.grad_f( np.ones( 1+neighbours_with_user.shape[0]))
				# return 
				# init_function_val[user] = grad_function_obj.f(x0)
				result= spg_obj.solve( x0, grad_function_obj.f, grad_function_obj.grad_f, self.project_positive, self.queue_length , sys.float_info.epsilon, self.max_iteration_mu_B )
				x = result['bestX']
				# if user%5 == 0:

				# 	plt.plot( result['buffer'])
					
				# end_function_val[user] = grad_function_obj.f(x)
				mu[user] = x[0]
				B[neighbours_with_user, user] = x[1:]
				# return	
		# plt.show()
		self.mu = mu
		self.B = B
		# plt.plot( init_function_val , 'r')
		# plt.plot( end_function_val , 'b')
		# plt.show()
		# print "likelihood function starts with "+ str( np.sum( init_function_val ) ) + " and ends with " + str( np.sum( end_function_val))
		 

	def solve_least_square(self,A,b):
		
		# print "A"+ str(A.shape)
		# print b.shape

		A_T_b = A.T.dot(b) 
		# print "shape "+ str(A_T_b.shape[0])
		# x=LA.solve( np.matmul(A.T,A)+self.lambda_least_square*np.eye( A_T_b.shape[0] ) , A.T.dot(b))
		mat = np.matmul(A.T,A) + (self.lambda_least_square*np.eye( A_T_b.shape[0] )) 
		# print "rank" + str(LA.matrix_rank(mat))
			# print mat
		# print "size" + str(mat.shape[0]) + "," + str(mat.shape[1])
		# for i in range(12):
		# 	if LA.norm(mat[:,i]) == 0 : 
		# 		print "col zero"
		x=LA.solve( mat , A_T_b)

		# x=LA.solve( self.lambda_least_square*np.eye( A_T_b.shape[0] ) , A.T.dot(b))
		# x=LA.solve( np.matmul(A.T,A), A.T.dot(b))
		# print x.shape
		# return x #**************
		return x[0],x[1:]
	

	def find_alpha_A(self):
		alpha = np.zeros(self.num_node)
		A = np.zeros((self.num_node, self.num_node ))

		#--------------------------------------------
		# self.alpha = alpha
		# self.A = A
		# return 
		#---------------------------------------------
		index={}
		for user in self.nodes :
                        #print('train shape', self.train.shape)
                        #print('return shape', np.where(self.train[:,0]==user).shape)
			index[user] = np.where(self.train[:,0]==user)[0]
		for user in self.nodes :  
			num_msg_user = index[user].shape[0]
			if num_msg_user > 0 :

				neighbours = np.nonzero(self.edges[:,user].flatten())[0]
				# neighbours = np.nonzero(self.edges[user,:])[0]
				num_nbr = neighbours.shape[0]
				g_user = np.zeros( ( num_msg_user, num_nbr ) )
				msg_user = self.train[index[user],2]
				nbr_no=0
				for nbr in neighbours :
					if index[nbr].shape[0] > 0 :
						user_msg_ind = 0
						time = 0 
						opn = 0 
						index_for_both = np.sort( np.concatenate((index[user], index[nbr])) )
						for ind in index_for_both : 
							user_curr , time_curr , sentiment = self.train[ind,:]
							if user_curr == user:
								opn = opn*np.exp(-self.w*(time_curr - time))
								g_user[user_msg_ind, nbr_no]=opn
								user_msg_ind = user_msg_ind + 1
								if user_msg_ind == num_msg_user:
									break
							else:
								opn = opn*np.exp(-self.w*(time_curr - time))+sentiment
							time = time_curr
					nbr_no = nbr_no + 1
				g_user = np.concatenate(( np.ones((msg_user.shape[0],1)) , g_user  ), axis = 1 )
				alpha[user], A[neighbours, user] = self.solve_least_square( g_user, msg_user )
		self.alpha = alpha
		self.A = A



		# print " diff of alpha "
		# print LA.norm(self.alpha - self.alpha_true)
		# print " diff of A "
		# print LA.norm( self.A - self.A_true)
	def generate_parameters(self):
		self.mu = rnd.uniform(low = 0 , high = 1 , size = self.num_node)
		self.alpha = rnd.uniform( low = -1 , high = 1 , size = self.num_node )
		self.A = np.zeros(( self.num_node , self.num_node ))
		self.B = np.zeros(( self.num_node , self.num_node ))
		for user in self.nodes:
			nbr = np.nonzero(self.edges[user,:])[0]
			# print "___user "+ str(user)+"_________"
			# print nbr
			# print "__________A_________"
			self.A[user ,  nbr ] = .1*rnd.uniform( low = -1 , high = 1 , size = nbr.shape[0] ) 
			# print self.A[user , :]
			# print "_____________________"
			self.B[user , np.concatenate(([user], nbr ))] = .1*rnd.uniform( size = nbr.shape[0] + 1  )
			# print self.B[user , :]
	def get_sigma_covar_only(self):
		self.find_alpha_A()
		return self.estimate_variance()

	def estimate_intensity_poisson(self):
		self.mu=np.zeros(self.num_node)
		self.B=np.zeros((self.num_node,self.num_node))
		time_interval=self.train[-1,1]-self.train[0,1]
		# print 'time',time_interval

		# print 'n_tr',self.num_train





		# print 
		for user in self.nodes : 
			self.mu[user] = float(np.where(self.train[:,0]==user)[0].shape[0])/time_interval
		# no_of_user_wt_zero_int=len([1 for v in self.mu if v == 0 ])

		# print 'sum',np.sum( self.mu)

		# print no_of_user_wt_zero_int


	def estimate_param(self,lamb=None, max_iter=None):
		if lamb!=None:
			self.lambda_least_square=lamb 
		
		
		# pass
		# estimate parameters
		# print "inside"
		# self.mu,self.B = self.find_mu_B()
		if self.int_generator=='Hawkes':
			self.find_mu_B(max_iter=max_iter)  
		else:
			self.estimate_intensity_poisson()
		
		self.find_alpha_A()
		self.var = self.estimate_variance()
		return {'A':self.A, 'B':self.B, 'alpha':self.alpha, 'mu':self.mu}
		# self.time_to_plot=[]
		# self.lambda_to_plot=[]
		# print self.mu
		# print self.alpha
		# print "____________E__________________________________________--______"
		# print self.edges
		# print "----------A-------------------------------------------------------"
		# print self.A
		# print "___________B___________________________________________________--------___------"
		# print self.B
		# print "----------nodes-------------"
		# print self.nodes

	def predict_opn_over_train( self ):
		time_old = 0
		opn = self.alpha 
		predicted_opn =  np.zeros( self.num_train)
		for msg, idx in zip(self.train, range(self.num_train)) : 
			user, time, sentiment = msg
			user = int( user )
			time_diff = time - time_old
			opn = self.alpha  + ( opn - self.alpha )*np.exp(-self.w*time_diff)+self.A[user]*sentiment 
			predicted_opn[idx] = opn[ user ]
			time_old = time
		return predicted_opn

	def estimate_variance( self ):
		predicted_opn = self.predict_opn_over_train()
		return np.mean(( predicted_opn - self.train[:,2])** 2 )
	def check_param_diff(self):
		pass
		# f1 = plt.figure()
		# plt.plot( self.mu_true, 'r')
		# plt.plot( self.mu , 'b')
		# # plt.show()

		# f1 = plt.figure()
		# plt.plot( self.alpha_true, 'r')
		# plt.plot( self.alpha , 'b')
		# # plt.show()

		# plt.show()

		# print "norm of A =" + str(LA.norm( self.A_true))
		# print "norm of A( estimated ) =" + str(LA.norm( self.A))
		# print "norm of diff of A =" + str(LA.norm( self.A_true- self.A))

		# print "norm of B =" + str(LA.norm( self.B_true))
		# print "norm of B( estimated ) =" + str(LA.norm( self.B))
		# print "norm of diff of B =" + str(LA.norm( self.B_true- self.B))

	def set_parameter(self, obj,flag_dict=False):
		if flag_dict:
			self.mu = obj['mu']
			self.alpha = obj['alpha'] 
			self.A = obj['A']
			self.B = obj['B']

		else:
			self.mu = obj.mu
			self.alpha = obj.alpha 
			self.A = obj.A
			self.B = obj.B
	def set_msg( self, flag_only_train = True,  msg_set = [] ):
		if flag_only_train :
			self.train = msg_set
		self.test = np.array([])
	def set_train( self, train_data ):
		self.train = train_data 
		self.num_train = self.train.shape[0]	
	def get_mean_intensity_of_msg( self, time, interval, timestamps ):
		tm_start=time - interval/2.0
		tm_end=time + interval/2.0
		return np.count_nonzero(np.logical_and( timestamps >= tm_start, timestamps <= tm_end))/(float(interval)*self.num_node)
	def get_mean_opinion_of_msg( self, time, interval, timestamps, opinions ):
		tm_start=time - interval/2.0
		tm_end=time + interval/2.0
		index = np.logical_and( timestamps >= tm_start, timestamps <= tm_end)
		return np.mean(opinions[index])
		
	def generate_empirical_mean_density_opinion( self, timestamps, opinions):
		Impirical_int = np.zeros(( self.num_window, timestamps.shape[0] ))
		Impirical_opn = np.zeros(( self.num_window, timestamps.shape[0] ))
		for window_index in range( self.num_window):
			for index in range( timestamps.shape[0]):
				interval=self.list_of_windows[window_index]
				Impirical_int[window_index,index] = self.get_mean_intensity_of_msg( timestamps[index], interval, timestamps)
				Impirical_opn[window_index,index] = self.get_mean_opinion_of_msg( timestamps[index], interval, timestamps, opinions)
		return Impirical_int, Impirical_opn

	def find_num_msg_poisson(self,time_span_input):
		counter=0
		for user in  self.nodes:
			counter+=int(self.mu[user]*time_span_input)
		self.num_msg_poisson=counter
	def predict(self, num_simulation, time_span_input):
		predict_test = np.zeros( (self.test.shape[0], num_simulation) )
		self.update_opn_int_history(-1) # train_int, train_opn 
		if self.int_generator=='Poisson':
			self.find_num_msg_poisson(time_span_input)
		msg_set = {}
		start=time.time()
		for msg_index in range(self.test.shape[0]): 
			predict_test[msg_index,:], msg_set_for_single_instance = self.predict_single_instance( self.test[msg_index], num_simulation, time_span_input)  
			msg_set[msg_index]= msg_set_for_single_instance
		print( 'time required',str(time.time()-start))
		mean_predict_test = np.mean( predict_test, axis = 1 ) 
		MSE_loss = get_MSE(mean_predict_test, self.test[:,2])
		FR_loss = get_FR(mean_predict_test, self.test[:,2])

		results = {}
		results['w']=self.w 
		results['v']=self.v
		results['MSE'] = MSE_loss 
		results['FR'] = FR_loss
		results['predicted']=predict_test
		results['true_target'] = self.test[:,2]
		results['msg_set']={} #  ms{} # g_set
		
		return results
	def mean_of_relevant_subset( self, curr_opn, list_of_windows, time):
		# timestamps = self.train[:,2]
		if list_of_windows.shape[0] > 1:
			print( 'Error, more than one window have been taken.')
		interval = list_of_windows[0]
		tm_start=time - interval/2.0
		tm_end=time + interval/2.0
		index = np.logical_and( self.timestamps >= tm_start, self.timestamps <= tm_end)

		# print 'number of msg in interval = ' +str(users_all.shape[0])
		# self.list_to_plot.append( users_all.shape[0])
		users = np.unique(self.users_all[index]).astype(int)
		# print users

		return np.mean(curr_opn[users])



	def update(self, msg_set_all, curr_time, required_running_intensity = False):
		# if required_running_intensity:
		# 	time_list=[]
		# 	intensity_list=[]

		time_array = msg_set_all[:,1]
		index_set = np.logical_and( (time_array > self.time_last) , (time_array <= curr_time) ) # check
		if np.count_nonzero(index_set) > 0 :
			time_last = self.time_last
			for user, time, sentiment in msg_set_all[index_set,:] :
				# print "msg time = "+ str( time )
				time_diff = time - time_last
				time_last = time
				user = int(user)
				self.curr_opn = self.alpha + (self.curr_opn - self.alpha) * np.exp( - self.w * time_diff) + self.A[user]*sentiment # define time diff 
				self.curr_int = self.mu + ( self.curr_int - self.mu) * np.exp( - self.v * time_diff) + self.B[user] # check it with abirda whether my curr msg will affect my intensity , perhaps it will
				# self.time_list.append( time)
				# self.estimated_int.append(np.mean(self.curr_int))
				# self.estimated_opn.append(np.mean(self.curr_opn))
				# self.estimated_opn.append( self.mean_of_relevant_subset( self.curr_opn, self.list_of_windows, time ) ) 
				# self.estimated_opn_exact.append( self.curr_opn[user])
			self.time_last = time

			#-----------------------------------------------------------
			# self.time_to_plot.append( time)
			# self.lambda_to_plot.append( np.mean(self.curr_int))
			#------------------------------------------------------------
		# print self.time_last
		# print " ------ updated by following number of msg "
		# print index_set.shape[0]
		# print " where number of train is "
		# print self.train.shape[0]
		
	def update_opn_int_history(self, curr_time): 
		# for all user, update their opn and int using msg from history upto current time and saves those values in curr_opn, curr_int
		# Also that case is not covered when test - time span exceeds last msg of train set 
		if curr_time == -1 :
			# print "initialized"
			# indicates it is called for initialization
			self.time_last= 0
			self.curr_opn = self.alpha
			self.curr_int = self.mu
			return



		if self.time_last == curr_time:
			return 

		# print "curr time "+str(curr_time)
		if self.time_last < self.train[-1,1]:
			# print "train"
			self.update( self.train , curr_time )
		if curr_time > self.train[-1,1]:
			# print "test"
			self.update( self.test , curr_time )

		# update for last interval of time , from time last to curr time ******************* 
		if self.time_last < curr_time : 

			# print " time of last msg " + str(self.time_last) + " for update is less than current time " + str( curr_time)
			time_diff = curr_time - self.time_last
			self.curr_opn = self.alpha + (self.curr_opn - self.alpha) * np.exp( - self.w * time_diff) 
			self.curr_int = self.mu + ( self.curr_int - self.mu) * np.exp( - self.v * time_diff) 
			self.time_last = curr_time

			
	def predict_single_instance( self, msg, num_simulation, time_span_input):
		user, time, sentiment = msg 
		# print " msg = " + str(user) + " " + str(time) + " " + str(sentiment)
		user = int(user)
		# get initial opn from history
		if time_span_input == 0 :
			start_time_sampling = time -  sys.float_info.epsilon
			# print " time span 0"
			# print " current time " + str( time )
			# print "sampling starts at " + str(start_time_sampling)
		else:	
			start_time_sampling = time - time_span_input
		if start_time_sampling < 0 : 
			print( "time to start smapling is less than zero , it equals ", str(start_time_sampling))
			start_time_sampling = 0 
			time_span_input = time
		# print " sampling start time " + str(start_time_sampling)

		self.update_opn_int_history( start_time_sampling)

		# print " opinion and intensity updated upto " + str( self.time_last) +  " where start time of sampling is " + str( start_time_sampling )
		# return
		# print "number of non zero entries in intensity values = " + str(np.count_nonzero( self.curr_int ))

		# plt.plot( self.curr_int )
		# plt.show()

		# self.plot_user_vs_msg()
		prediction_array = np.zeros( num_simulation )
		msg_set_all_simul = {}
		for simulation_no in range(num_simulation):
			# sample using opinions obtained in previous step
			if self.int_generator=='Hawkes':
				msg_set, last_opn_update, last_int_update =  self.simulate_events(time_span_input , self.curr_int, self.curr_opn,  self.A, self.B)
			else:
				if self.int_generator=='Poisson':
					# print 'true'
					msg_set, last_opn_update, last_int_update =  self.simulate_events_poisson(time_span_input,self.mu,self.curr_opn,self.A)
				else:
					print( 'intensity generator not specified'	)
			msg_set_all_simul[ simulation_no ] = msg_set
			# find_num_msg_plot( msg_set, self.num_node, self.curr_int)
			# all three above variable store times assuming start time as 0 whereas start time is actually "time"
			# therefore before using them they must be corrected
			# print "the number of msg in 4 hour of sampling is = " + str(msg_set.shape[0])
			
			if msg_set.shape[0] > 0 :
				msg_set[:,1] += start_time_sampling

			last_int_update[:,0] += start_time_sampling
			last_opn_update[:,0] += start_time_sampling			

			# curr_int = np.zeros( self.num_node )
			# curr_opn = np.zeros( self.num_node )
			# for node in self.nodes:
			# 	curr_int[ node ] = last_int_update[node,1]*np.exp(-self.v*(time - last_int_update[node,0]))
			# 	curr_opn[ node ] = last_opn_update[node,1]*np.exp(-self.w*(time - last_opn_update[node,0]))
			prediction_array[simulation_no] = self.predict_from_events( self.alpha[user], self.A[:,user] , last_opn_update[user,:],  msg_set, user , time ) # or perhaps next one 
		return prediction_array, msg_set_all_simul #, curr_int, curr_opn

	# def simulate_events(self,a):

	def simulate_events_poisson(self,time_span,mu,alpha,A): # Alpha is curr opn , but mu is initial at time 0 
		
		time_init = np.zeros((self.num_node,1))
		opn_update = np.concatenate((time_init, alpha.reshape(self.num_node , 1 )), axis=1)
		int_update = np.zeros((self.num_node,2))
		msg_set = []
		if time_span == 0:
			# print 'Invalid option'
			return np.array(msg_set), opn_update, int_update
		# num_msg=int(math.ceil(time_span*np.sum(mu)))
		msg_times=np.zeros((self.num_msg_poisson,2))
		counter=0
		for user in  self.nodes:
			n_msg=int(mu[user]*time_span)
			if n_msg!=0:
				msg_times[counter:counter+n_msg,0]=user
				# print 'n_msg',n_msg,'   counter',counter,'   counter+n_msg',int(counter+n_msg)
				msg_times[counter:counter+n_msg,1]=rnd.uniform(low=0,high=time_span,size=n_msg) # .reshape(n_msg,1)
				counter+=n_msg
		for itr_no in range(self.num_msg_poisson):
			index = np.argmin(msg_times[:,1])
			u=int(msg_times[index,0])
			t_new=msg_times[index,1]
			if t_new==float('inf'):
				print( 'check line 884 in simulate event poisson')
				break
			msg_times[index,1] = float('inf')
			t_old,x_old = opn_update[u,:]
			x_new=alpha[u]+(x_old-alpha[u])*np.exp(-self.w*(t_new-t_old))
			# opn_update_time[u]=t_new
			# opn_update_val[u]=x_new
			opn_update[u,:]=np.array([t_new,x_new])
			m = rnd.normal( x_new , math.sqrt(self.var) )
			msg_set.append(np.array([u,t_new,m]))
			
			for nbr in np.nonzero(self.edges[u,:])[0]:
				# print 
				# print " ------------for nbr " + str(nbr) + "-------------------------"
				t_old,x_old=opn_update[nbr]
				x_new = alpha[nbr] + ( x_old - alpha[nbr] )*np.exp(-self.w*(t_new-t_old)) + A[u,nbr]*m
				opn_update[nbr]=np.array([t_new,x_new])
		return np.array(msg_set) , opn_update, int_update

	def simulate_events(self, time_span,mu,alpha,A,B,flag_check_num_msg = False,max_msg=0,return_only_opinion_updates= False,var=None,p_exo=None,p_dist=None,noise=None):
		# start_time = time.time()
		#---------------------self variables used in this module ---------
		# edges
		# w
		# v 
		#-----------------------------------------------------------------
		# plt.plot( mu )
		# plt.show()
		# print a
		# return np.array([5])
		# max_msg=10000
		if var!=None:
			self.var=var

		if p_exo!=None:
			msg_end_list=[]


		time_init = np.zeros((self.num_node,1))
		opn_update = np.concatenate((time_init, alpha.reshape(self.num_node , 1 )), axis=1)
		int_update =  np.concatenate((time_init, mu.reshape( self.num_node , 1 )), axis=1)
		
		msg_set = []

		if time_span == 0:
			# opn_update[:,0] = self.time_last
			# int_update[:,0] = self.time_last # check whether such assignment works

			return np.array(msg_set), opn_update, int_update

		tQ=np.zeros(self.num_node)
		# print 'zero',np.count_nonzero(mu)
		for user in self.nodes:
			# if mu[user] == 0 :
			# 	print "initial intensity  = zero "
			tQ[user] = self.sample_event( mu[user] , 0 , user, time_span, mu[user] ) 
			# tQ[user] = rnd.uniform(0,T)
		# Q=PriorityQueue(tQ) # set a chcek on it
		# print "----------------------------------------"
		# print "sample event starts"

		# print 'min',tQ.min()
		# print 'max',tQ.max()
		t_new = 0
		#--------------------------------------------------
		if flag_check_num_msg:
			num_msg = 0
		# num_msg=0
		#--------------------------------------------------
		while t_new < time_span:
			u = np.argmin(tQ)
			t_new = tQ[u]
			tQ[u] = float('inf')
			# t_new,u=Q.extract_prior()# do not we need to put back t_new,u * what is this t_new > T 
			# u = int(u)

			# print " extracted user " + str(u) + "---------------time : " + str(t_new)
			# t_old=opn_update_time[u]
			# x_old=opn_update_val[u]
			t_old,x_old = opn_update[u,:]
			x_new=alpha[u]+(x_old-alpha[u])*np.exp(-self.w*(t_new-t_old))
			# opn_update_time[u]=t_new
			# opn_update_val[u]=x_new
			opn_update[u,:]=np.array([t_new,x_new])
			if p_exo!=None:
				# method 2 
				seed=rnd.uniform(0,1,1)
				if seed<p_exo:
					msg_end_list.append(False)
					if p_dist!=None:
						m=p_dist() # rnd.beta(5,1,1)
					if p_dist==None and noise==None:
						m= rnd.normal( rnd.normal(0,1) , math.sqrt(.1))
					if noise!=None: # indicates you are purturbing 
						m=rnd.normal( x_new , math.sqrt(self.var) )+noise()
				else:
					msg_end_list.append(True)
					m = rnd.normal( x_new , math.sqrt(self.var) )
				# method 1 
				# seed=rnd.uniform(0,1,1)
				# if seed<p_exo:
				# 	msg_end_list.append(False)
				# 	if p_dist!=None:
				# 		m=p_dist() # rnd.beta(5,1,1)
				# 	if noise!=None:
				# 		m=rnd.normal( x_new , math.sqrt(self.var) )+noise()
				# else:
				# 	msg_end_list.append(True)
				# 	m = rnd.normal( x_new , math.sqrt(self.var) )
			else:
				m = rnd.normal( x_new , math.sqrt(self.var) )
			# print 'num msg', num_msg
			# num_msg+=1

			
			# update neighbours
			for nbr in np.nonzero(self.edges[u,:])[0]:
				# print 
				# print " ------------for nbr " + str(nbr) + "-------------------------"
				# change above 
				t_old,lda_old = int_update[nbr]
				lda_new = mu[nbr]+(lda_old-mu[nbr])*np.exp(-self.v*(t_new-t_old))+B[u,nbr]# use sparse matrix
				int_update[nbr,:]=np.array([t_new,lda_new])
				t_old,x_old=opn_update[nbr]
				x_new = alpha[nbr] + ( x_old - alpha[nbr] )*np.exp(-self.w*(t_new-t_old)) + A[u,nbr]*m
				opn_update[nbr]=np.array([t_new,x_new])

				# print " updated int " + str(lda_new) + " ------------ updated opinion -----" + str(x_new)
				t_nbr=self.sample_event(lda_new,t_new,nbr, time_span, mu[nbr] )
				# print " update next event time of " + str( nbr ) + "  as " + str(t_nbr)
				tQ[nbr]=t_nbr

			msg_set.append(np.array([u,t_new,m]))
			if flag_check_num_msg == True:
				# print 'num msg', num_msg 
				num_msg = num_msg + 1 
				if num_msg > max_msg :
					break

		if p_exo!=None: 
			return np.array(msg_set) , np.array(msg_end_list, dtype=bool)

		if return_only_opinion_updates:
			return opn_update
		else:
			#print 'num_msg=',len(msg_set)
			return np.array(msg_set) , opn_update, int_update
	def sample_event(self,lda_init,t_init,user, T,mu ): # to be checked
		lda_old=lda_init
		t_new= t_init
		 
		
		# print "------------------------"
		# print "start tm "+str(t_init) + " --- int --- " + str(lda_init)
		# print "------------start--------"
		# itr = 0 
		# print 'we are starting with intensity ', lda_init
		# max_iter=10
		# itr=0
		while t_new< T : 
			u=rnd.uniform(0,1)
			# if lda_old == 0:
			# 	print "itr no " + str(itr)
			delta_t =math.log(u)/lda_old 
			# print math.log(u)
			# print lda_old
			t_new -= delta_t
			# print 'time is decreased by ',delta_t
			# print "new time ------ " + str(t_new)
			lda_new =mu + (lda_init-mu)*np.exp(-self.v*(t_new-t_init))
			# print "new int  ------- " + str(lda_new)
			d = rnd.uniform(0,1)
			# print "d*upper_lda : " + str(d*lda_upper)
			if d*lda_old < lda_new  :
				break
			else:
				lda_old = lda_new
			# itr += 1
			# print 'current time ' ,t_new
			# itr+=1
			# if itr>max_iter:
			# 	break

		return t_new # T also could have been returned

	def predict_from_events( self, alpha, A , last_opn_update,  msg_set, user , time ): # confirm from abir da whether to send alpha or curr opn variable of self 
		
		# print "inside predict from events"
		
		time_last, opn  = last_opn_update 
		if time_last == time:
			# print " user's opinion is already updated"
			return opn
		
		# print "user's last opinion " + str(opn)
		# print "uers last time " + str( time_last)
		
		if msg_set.shape[0] == 0: # check
			time_diff = time - time_last
			# print " time difference "+ str( time_diff)
			# print "previous opn " + str( opn ) 
			opn = alpha + (opn - alpha )*np.exp(-self.w*time_diff)
			# print " opinion after updating for epsilon time is " + str( opn )
			return opn

		time_array = msg_set[:,1]	

		# print " time of msg set "
		# print time_array

		ind = np.logical_and((time_array > time_last ), ( time_array < time)) 
		
		# print "selected index = "
		# print ind 
		# print "------------------"
		
		for user_curr, time_curr, sentiment in msg_set[ ind,:]: 

			# print " curr user "+ str( user_curr)

			if self.edges[user,int(user_curr)]>0 :

				# print " selected user to update my opn = " + str( user_curr)

				time_diff = time_curr - time_last

				# print " time diff = "+str( time_diff)
				# print "previous opn = " + str( opn )


				opn = alpha + (opn - alpha )*np.exp(-self.w*time_diff)+A[int(user_curr)]*sentiment
				
				# print " alpha " + str( alpha)
				# print " influence "+ str( A[ int(user_curr)])

				# print " curr opn " + str( opn )
				# print " user's opinion is updated by other's msg as " + str(opn)
				time_last = time_curr

			# else:
				# print " not selected "
		
		# print " out of loop " 

		time_diff = time - time_last
		# print " curr time diff " + str( time_diff )
		if time_diff > 0 :
			opn = alpha + (opn - alpha )*np.exp(-self.w*time_diff)
		# print "updated opinion " + str( opn )
		# print " final opn = " + str(opn)
		return opn
	# def check_validity_of_param(self):
	# 	if np.count_nonzero(self.mu) == 0 :
	# 		print "all initial intensity are zero"
	# 	if np.count_nonzero(self.alpha) == 0 :
	# 		print "all initial opinion are zero"
	# 	if np.count_nonzero(self.A) == 0 :
	# 		print "all initial opinion influence are zero"
	# 	if np.count_nonzero(self.B) == 0 :
	# 		print "all initial intensity influence are zero"
def tuning_slant( file_prefix, time_span_input , list_of_w, list_of_v , list_of_lambdas , num_simulation ):
	directory = "../slant_tuning_result"
	if not os.path.exists(directory):
		os.makedirs(directory)
	file_to_read =  '../Cherrypick_others/Data_opn_dyn_python/' + file_prefix + '_10ALLXContainedOpinionX.obj'
	file_to_write = directory + '/' + file_prefix + 'res.slant.tuning'
	obj = load_data( file_to_read )
	result_list = []

	index = 0 
	for w in list_of_w : 
		for v in list_of_v:
			for lamb in list_of_lambdas : 
				# init slant
				start = time.time()
				slant_obj = slant( obj=obj, init_by='object' , data_type = 'real' , tuning = True , tuning_param = [ w,v,lamb ])
				# slant_obj.estimate_param( )
				res_obj = {} # slant_obj.predict( num_simulation = num_simulation  , time_span_input = time_span_input )
				# print 'w : ' + str(w) + ', v : ' + str(v) + ' lambda : ' + str( lamb )
				# print ' single instanciation and estimation takes ' + str( time.time() - start )							
				res_obj['w'] = w
				res_obj['v'] = v
				res_obj['lamb'] = lamb 
				result_list.append( res_obj )
				save( result_list , file_to_write + '.' + str(index)  )
				index += 1 
	save( result_list , file_to_write )
def check_opinion_range( file_prefix):
	path = '../Cherrypick_others/Data_opn_dyn_python/'
	file_suffix = '_10ALLXContainedOpinionX.obj'
	obj = load_data( path+file_prefix+file_suffix)
	opn = np.concatenate( ( obj.train[:,2] , obj.test[:,2]  ) )
	details = {}
	details['name'] = file_prefix
	details['mean'] = np.mean(opn)
	details['min'] = np.min( opn )
	details['max'] =  np.max(opn)
	details['number of zero'] = opn.shape[0] - np.count_nonzero(opn)
	details['fraction of zero'] = float(details['number of zero'])/opn.shape[0]
	# print 'Dataset ' + file_prefix
	# print 'mean ' + str( details['mean'])
	# print 'min ' + str( details['min'])
	# print 'max ' + str( details['max'])
	# print 'number of 0 ' + str( details['number of zero'])
	# print 'fraction of zero ' + str(  details['fraction of zero'])
	# print 'norm of opn ' + str( LA.norm(opn))

def eval_using_slant( file_prefix, time_span_input_list, method, num_simulation, list_of_windows = None , w=None,v=None,lamb = None, file_to_write_result=None):
	# time_span_input_slant = # ***************************
	# print ' file name ' + file_prefix 
	path = '../Cherrypick_others/Data_opn_dyn_python/'
	file_suffix = '_10ALLXContainedOpinionX.obj'
	file_to_read_obj = path + file_prefix + file_suffix 
	# if file_to_write_result == None:
	# 	file_to_write_result = '../result/' + file_prefix + '.lamb0'+ str(lamb)+'.res.slant'
	if w==None:
		w=10
	if v==None:
		v=10
	if lamb==None:
		lamb=1
	directory = "../result"
	if not os.path.exists(directory):
		os.makedirs(directory)
		
	obj = load_data( file_to_read_obj) 
	start = time.time()
	slant_obj = slant( obj=obj, init_by='object' , data_type = 'real', list_of_windows = list_of_windows, tuning=True, tuning_param=[w,v,lamb] ) 
	# obj.train = np.concatenate((obj.train, obj.test),axis=0)
	# plt.plot(obj.train[:,1])
	# plt.show()
	# return 
	slant_obj.estimate_param()
	# print ' single instanciation and estimation takes ' + str( time.time() - start )
	result_list_inner = []
	for time_span_input in time_span_input_list :

		start = time.time()
		# print ' time span input ' + str( time_span_input )
		if time_span_input==0:
			result_obj = slant_obj.predict( num_simulation =1 , time_span_input = time_span_input )
		else:
			result_obj=slant_obj.predict(num_simulation=num_simulation,time_span_input=time_span_input)
		# print ' single prediction takes ' + str( time.time() - start )
		msg =dict( result_obj['msg_set'])
		del result_obj['msg_set']
		save(msg,file_to_write_result+'t'+str(time_span_input)+'.res.slant.msg')
		save(result_obj, file_to_write_result+'t'+str(time_span_input)+'.res.slant')
		# result_obj = []
		result_list_inner.append( result_obj )
		
	return result_list_inner

def get_best_lambda_idx( list_of_res):
	list_of_MSE=[]
	for res in list_of_res:
		list_of_MSE.append(res['MSE'])
	return np.argmin( np.array(list_of_MSE))

