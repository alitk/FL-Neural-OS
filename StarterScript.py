import os
from utils.options import args_parser


#example 
#python StarterScript.py --client --idx 5 --verbose --num_users 10 --iid --local_bs 60 --base_file ./where/to/read/base_model --local_dir ./where/to/save
#python StarterScript.py --server --local_dir where/to/read/ --saveto where/to/save

#if you want to delete model after merging them
#python StarterScript.py --server --local_dir where/to/read/ --saveto where/to/save --rm_local

args = args_parser()


assert (args.client or args.server), "specify model e.g. --client or --server."

if args.client and not(args.server):

	cmd_run ="python client.py {} --idx {} {} --num_users {} {} --local_bs {} --base_file {} --local_dir {}".format(args.new*'--new',
																										 args.idx, args.verbose*'--verbose',
																										  args.num_users, args.iid*'--iid',
																										   args.local_bs, args.base_file, args.local_dir)
elif not(args.client) and  args.server:
	cmd_run="python server.py"
	
	
else:
	"Model should be either client or server"

print(cmd_run)
os.system(cmd_run)
