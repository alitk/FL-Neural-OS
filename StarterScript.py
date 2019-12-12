import os
from utils.options import args_parser


#example 
#python StarterScript.py --new --idx 5 --verbose --num_users 10 --iid --local_bs 60 --base_file ./LocalModel/local0.pth --client
#python StarterScript.py --server

#if you want to delete model after merging them
#python StarterScript.py --server

args = args_parser()


assert (args.client or args.server), "specify model e.g. --client or --server."

if args.client and not(args.server):

	cmd_run ="python client.py {} --idx {} {} --num_users {} {} --local_bs {} --base_file {}".format(args.new*'--new',
																										 args.idx, args.verbose*'--verbose',
																										  args.num_users, args.iid*'--iid',
																										   args.local_bs, args.base_file)
elif not(args.client) and  args.server:
	cmd_run="python server.py"
	
	
else:
	"Model should be either client or server"

print(cmd_run)
os.system(cmd_run)
