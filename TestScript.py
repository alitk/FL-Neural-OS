num_users=10
import os

for i in range(num_users):
	os.system("python client.py --new t --idx {} --verbose --num_users {} --iid --local_bs 60".format(i, num_users))

os.system("python server.py --num_users {}".format(num_users))