import getopt 
import time
#import matplotlib.pyplot as plt
import numpy as np
from slant import *
from myutil import *




def get_lambda_new():


	exp_dict = { '0':'vary_time', '1':'vary_frac', '2':'san_test', '3':'vary_train'}
	files_dict = {'ba':'barcelona', 	'br' : 'british_election', 	'j' : 'jaya_verdict', 'J':'juventus'  , 'T':'delhi_election' }
	method_dict= { 'acpk' : 'cherrypick_a', 'ecpk':'cherrypick_e', 'tcpk':'cherrypick_t','Robust_cherrypick' : 'robust_ht', \
	'cherrypick':'cherrypick_d', 'slant':'slant','robust_lasso':'robust_lasso', 'soft_thresholding':'soft_thres','huber_regression':'huber_loss'}

	# lamb_dict = {}
	# for exp_no in range(4):
	# 	lamb_dict[ exp_dict[str(exp_no)] ] = {}
	# 	for file_key in files_dict:
	# 		lamb_dict[ exp_dict[str(exp_no)] ][ files_dict[file_key] ]={}
	# 		for method_key in method_dict:
	# 			lamb_dict[ exp_dict[str(exp_no)] ][ files_dict[file_key] ][ method_dict[method_key] ] = {}

	lamb_file = 'lambda'
	lamb_dict = load_data(lamb_file)
		
	for exp_no in range(1):
		for ext in ['acpk','ecpk','tcpk'] :
			fname = '../../../current/exp'+str(exp_no+1)+'/Result_Slant_Short/'+ext+'.txt'
			with open( fname ,'r') as f :
				line = f.readline()
				while line:
					if line.split(',')[0][0] in ['b','j','J','T']:

						if exp_no == 0 :
							if len( line.split(',')	) == 2:
								l = [ float(line.split(',')[1]) ] * 6 
							else:
								l = [ float(s) for s in line.split(',')[1:] ]
						# if exp_no == 1:
						# 	if len( line.split(',')	) == 2:
						# 		l = [ float(line.split(',')[1]) ] * 5 
						# 	else:
						# 		l = [ float(s) for s in line.split(',')[1:] ]
						# 	l.insert(2, lamb_dict[ exp_dict[str(exp_no)] ][ files_dict[line.split(',')[0]] ][ method_dict(ext) ][2] )
						# if exp_no == 2:
						# 	l = lamb_dict[ exp_dict['0'] ][ files_dict[line.split(',')[0]] ][ method_dict(ext) ][2]
						# if exp_no == 3 : 
						# 	if len( line.split(',')	) == 2:
						# 		l = [ float(line.split(',')[1]) ] * 5 
						# 	else:
						# 		l = [ float(s) for s in line.split(',')[1:] ]

						lamb_dict[ exp_dict[str(exp_no)] ][ files_dict[line.split(',')[0]] ][ method_dict[ext] ] = l
					line = f.readline()

	print( lamb_dict )
	save( lamb_dict, lamb_file )
	




def get_lambda_old():

	file_dict = {'barcelona':'barca',	'british_election' : 'british_election','jaya_verdict' : 'jaya_verdict', 'juventus' : 'JuvTwitter' , 'delhi_election' : 'Twitter' }
	
	lamb_dict = load_data('lambda')
	# print(lamb_dict)
	# return	
	param={}
	param['lambda']=lamb_dict
	param['threshold_fr']= {}
	for file_name in file_dict.keys():
		res = load_data( '../../../current/exp1/Result_Slant_Short/res.'+ file_dict[file_name])
		# print(res)
		param['threshold_fr'][file_name]=res['threshold']

	print(param)
	save(param, 'hyper_params')
	return

	for exp_no in range(4):
		for file_name in file_dict.keys():
			
			print( '**********************')
			

			if exp_no == 0:
				res = load_data( '../../../current/exp'+str(exp_no+1)+'/Result_Slant_Short/res.'+ file_dict[file_name])
				method_dict= { 'Robust_cherrypick' : 'robust_ht', 'cherrypick':'cherrypick_d', 'slant':'slant',\
				'robust_lasso':'robust_lasso', 'soft_thresholding':'soft_thres','huber_regression':'huber_loss'}
				for method_key  in method_dict:	
					# print(method, ':',res['lambda']['hawkes'][ method_key ])
					lamb_dict['vary_time'][file_name][ method_dict[method_key]] = list(res['lambda']['hawkes'][method_key])
			

			if exp_no == 1 :
				res = load_data( '../../../current/exp'+str(exp_no+1)+'/Result_Slant_Short/'+ file_dict[file_name]+'.res')
				# print(res)	

				method_dict = {'acpk':'cherrypick_a','cpk':'cherrypick_d','ecpk':'cherrypick_e','rcpk':'robust_ht','tcpk':'cherrypick_t'}
				for method_key in method_dict:
					lamb_dict['vary_frac'][file_name][ method_dict[method_key]] = list(res['lambda'][method_key])
					if method_key in ['acpk','ecpk','tcpk']:
						lamb_dict['vary_frac'][file_name][ method_dict[method_key]].insert(2, lamb_dict['vary_time'][file_name][method_dict[method_key]][2])


			if exp_no == 3 : 
				res = load_data( '../../../current/exp'+str(exp_no+1)+'/Result_Slant_Short/'+ file_dict[file_name]+'.res')
				# print(res)
				method_dict = {'tcpk':'cherrypick_t','ecpk':'cherrypick_e','acpk':'cherrypick_a'}
				for method_key in method_dict:
					lamb_dict['vary_train'][file_name][ method_dict[method_key]] = list(res['lambda'][method_key])

				method_dict = {'robust_lasso':'robust_lasso', \
				'soft_thresholding':'soft_thres','huber_regression':'huber_loss'}
				for method_key in method_dict:
					lamb_dict['vary_train'][file_name][ method_dict[method_key]] = [float(lamb_dict['vary_time'][file_name][method_dict[method_key]][2]) ] * 6


				method_dict = {'Robust_cherrypick' : 'robust_ht', 'cherrypick':'cherrypick_d', 'slant':'slant'}
				res = load_data( '../../../current/exp1/Result_Slant_Short/res.'+ file_dict[file_name])
				for method_key in method_dict:
					lamb_dict['vary_train'][file_name][ method_dict[method_key]] = [res['lambda']['poisson'	][method_key][2] ] * 6



			# for method_key  in method_dict:
			# 	# print(method, ':',res['lambda']['hawkes'][ method_key ])
			# 	lamb_dict['vary_time'][file_name][ method_dict[method_key]] = res['lambda']['hawkes'][method_key]
	print(lamb_dict['vary_time']) 

	print('\n\n')
	print(lamb_dict['vary_frac'])

	print('\n\n')

	print(lamb_dict['vary_train'])

	print('\n\n')

	save( lamb_dict, 'lambda')



def main():

	files = [ 'barcelona', 'british_election' , 'jaya_verdict', 'juventus', 'delhi_election' ]
	methods = ['cherrypick_a', 'cherrypick_d', 'cherrypick_e', 'cherrypick_t', 'robust_ht', 'robust_lasso', 'soft_thres', 'slant']
	times = [0.0,0.1,0.2,0.3,0.4,0.5]
	expnames = ['vary_time','vary_frac','sanitize_test','vary_train']
	


	# get_lambda_new()
	get_lambda_old()
	# file_name, t, method, frac , exp = parse_command_line_input( files, methods, times, expnames )
	
	# get_lambda( file_name , method, exp )

if __name__== "__main__":
  main()



